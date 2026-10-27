"""Keep 1,000 plausible, diverse Wuji grasp trajectories with per-object balance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _trajectory_features(data, indices):
    wrist = np.asarray(data["wrist_trajectory"][indices], dtype=np.float64)
    object_pose = np.asarray(data["object_pose_world_trajectory"][indices], dtype=np.float64)
    joints = np.asarray(data["palm_joint_position_trajectory"][indices], dtype=np.float64)
    object_rotation = Rotation.from_quat(object_pose[..., 3:7].reshape(-1, 4))
    wrist_rotation = Rotation.from_quat(wrist[..., 3:7].reshape(-1, 4))
    relative_rotation = (object_rotation.inv() * wrist_rotation).as_rotvec().reshape(len(indices), -1, 3)
    relative_position = object_rotation.inv().apply(
        (wrist[..., :3] - object_pose[..., :3]).reshape(-1, 3)
    ).reshape(len(indices), -1, 3)
    return relative_position / 0.04, relative_rotation / 0.7, joints / 0.5


def _pairwise_distance(features):
    count = len(features[0])
    distance = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            terms = [np.mean((feature[i] - feature[j]) ** 2) for feature in features]
            distance[i, j] = distance[j, i] = np.sqrt(sum(terms))
    return distance


def _select_group(data, indices, quota, quality):
    if len(indices) <= quota:
        return indices.tolist()
    features = _trajectory_features(data, indices)
    distance = _pairwise_distance(features)
    local_quality = quality[indices]
    first = int(np.argmin(local_quality))
    selected = [first]
    min_distance = distance[first].copy()
    min_distance[first] = -1.0
    while len(selected) < quota:
        # Break ties by the smoother trajectory to make the result repeatable.
        candidates = np.flatnonzero(min_distance == min_distance.max())
        next_index = int(candidates[np.argmin(local_quality[candidates])])
        selected.append(next_index)
        min_distance = np.minimum(min_distance, distance[next_index])
        min_distance[selected] = -1.0
    return indices[np.asarray(selected)].tolist()


def filter_dataset(source: Path, output: Path, target_count: int = 1000) -> dict:
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    count = len(data["object_label"])
    if count < target_count:
        raise ValueError(f"dataset has {count} rows, fewer than target {target_count}")

    wrist = np.asarray(data["wrist_trajectory"], dtype=np.float64)
    joints = np.asarray(data["palm_joint_position_trajectory"], dtype=np.float64)
    object_pose = np.asarray(data["object_pose_world_trajectory"], dtype=np.float64)
    quat_norm = np.linalg.norm(wrist[..., 3:7], axis=-1)
    step = np.linalg.norm(np.diff(wrist[..., :3], axis=1), axis=-1).max(axis=1)
    acceleration = np.linalg.norm(np.diff(wrist[..., :3], n=2, axis=1), axis=-1).max(axis=1)
    valid = (
        np.isfinite(wrist).all(axis=(1, 2))
        & np.isfinite(joints).all(axis=(1, 2))
        & np.isfinite(object_pose).all(axis=(1, 2))
        & (quat_norm.min(axis=1) > 0.98)
        & (quat_norm.max(axis=1) < 1.02)
        & (step <= 0.085)
        & (acceleration <= 0.12)
    )
    if valid.sum() < target_count:
        raise ValueError(f"only {valid.sum()} valid rows remain, fewer than target {target_count}")

    labels, inverse, counts = np.unique(data["object_label"], return_inverse=True, return_counts=True)
    valid_counts = np.bincount(inverse[valid], minlength=len(labels))
    if np.any(valid_counts == 0):
        raise ValueError("at least one object has no valid trajectories after quality filtering")
    # Keep each object's quota at floor/ceil(target / objects), spreading the
    # extra rows first across objects with the largest available sample pools.
    base, remainder = divmod(target_count, len(labels))
    quotas = np.minimum(valid_counts, base)
    remaining = target_count - int(quotas.sum())
    capacity_order = sorted(range(len(labels)), key=lambda i: (-valid_counts[i], str(labels[i])))
    while remaining:
        progressed = False
        for i in capacity_order:
            if quotas[i] < valid_counts[i]:
                quotas[i] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("cannot meet target count with available object rows")

    quality = step / 0.075 + acceleration / 0.10
    selected = []
    for group_index, label in enumerate(labels):
        group = np.flatnonzero((inverse == group_index) & valid)
        selected.extend(_select_group(data, group, int(quotas[group_index]), quality))
    selected = np.asarray(sorted(selected), dtype=np.int64)
    if len(selected) != target_count:
        raise AssertionError(f"selected {len(selected)} rows, expected {target_count}")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{
        key: values[selected] if values.ndim > 0 and values.shape[0] == count else values
        for key, values in data.items()
    })
    kept_counts = np.bincount(inverse[selected], minlength=len(labels))
    return {
        "input_rows": count,
        "valid_rows": int(valid.sum()),
        "rejected_quality_rows": int((~valid).sum()),
        "output_rows": len(selected),
        "objects": len(labels),
        "min_per_object": int(kept_counts.min()),
        "max_per_object": int(kept_counts.max()),
        "object_counts": {str(label): int(n) for label, n in zip(labels, kept_counts)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("datasets/wuji_grasp.npz"))
    parser.add_argument("--output", type=Path, default=Path("datasets/wuji_grasp_filtered_1000.npz"))
    parser.add_argument("--target-count", type=int, default=1000)
    args = parser.parse_args()
    summary = filter_dataset(args.input, args.output, args.target_count)
    print(
        f"Saved {summary['output_rows']} trajectories for {summary['objects']} objects to {args.output}; "
        f"quality rejected {summary['rejected_quality_rows']}, per-object kept "
        f"{summary['min_per_object']}–{summary['max_per_object']}"
    )


if __name__ == "__main__":
    main()
