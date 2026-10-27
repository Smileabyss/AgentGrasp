"""Evaluate both stages of RGB-D point-cloud completion and canonicalization."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from perception.rgbd_frame import prepare_two_stage_clouds
from perception.train_rgbd_box_estimator import build_pointr_model, sample_points_fps


POINTR_ROOT = Path(__file__).resolve().parents[1] / "external" / "PoinTr"
sys.path.insert(0, str(POINTR_ROOT))
from extensions.chamfer_dist import ChamferDistanceL1


def _icp_transform(source, target, max_iterations=50):
    """Estimate row-vector rigid transform source @ R + t ~= target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_axes = np.linalg.eigh(np.cov(source.T))[1][:, ::-1]
    target_axes = np.linalg.eigh(np.cov(target.T))[1][:, ::-1]
    tree = cKDTree(target)
    best = None

    for signs in product((-1.0, 1.0), repeat=3):
        candidate = source_axes @ np.diag(signs) @ target_axes.T
        if np.linalg.det(candidate) < 0.0:
            continue
        rotation = candidate
        translation = target_center - source_center @ rotation
        previous_error = float("inf")
        for _ in range(max_iterations):
            transformed = source @ rotation + translation
            distances, indices = tree.query(transformed, k=1)
            keep = distances <= np.percentile(distances, 85)
            matched_source = transformed[keep]
            matched_target = target[indices[keep]]
            source_mean = matched_source.mean(axis=0)
            target_mean = matched_target.mean(axis=0)
            u, _, vt = np.linalg.svd((matched_source - source_mean).T @ (matched_target - target_mean))
            update = u @ vt
            if np.linalg.det(update) < 0.0:
                u[:, -1] *= -1.0
                update = u @ vt
            offset = target_mean - source_mean @ update
            rotation = rotation @ update
            translation = translation @ update + offset
            error = float(np.mean(distances[keep]))
            if abs(previous_error - error) < 1e-8:
                break
            previous_error = error

        transformed = source @ rotation + translation
        distances, _ = tree.query(transformed, k=1)
        score = float(np.mean(np.sort(distances)[:max(3, int(0.85 * len(distances)))]))
        if best is None or score < best[0]:
            best = (score, rotation.copy(), translation.copy())
    return best[1], best[2], best[0]


def _rotation_error_degrees(estimated, target):
    relative = estimated @ target.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _load_sample(dataset_dir, global_index):
    shard_paths = sorted(Path(dataset_dir).glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError("No shard_*.npz files found in {}".format(dataset_dir))
    offset = 0
    for path in shard_paths:
        with np.load(path) as shard:
            count = len(shard["points_camera"])
            if global_index < offset + count:
                index = global_index - offset
                return {
                    key: shard[key][index].copy()
                    for key in ("depth", "object_mask", "points_camera", "pose_camera")
                }
        offset += count
    raise IndexError("sample index {} is outside [0, {})".format(global_index, offset))


def _show(
    partial, estimated_target, stage1_prediction, object_target,
    stage2_prediction, output_path, show, stage1_chamfer, stage2_chamfer,
    sample_index,
):
    fig = plt.figure(figsize=(18, 6))
    axes = (
        (partial, None, "Partial input\n(estimated frame)", "estimated"),
        (stage1_prediction, estimated_target,
         "Stage 1: completion in estimated frame\nChamfer L1 {:.6f} m".format(stage1_chamfer), "estimated"),
        (stage2_prediction, object_target,
         "Stage 2: canonicalization to object frame\nChamfer L1 {:.6f} m".format(stage2_chamfer), "object"),
    )
    for index, (prediction, target, title, frame_label) in enumerate(axes):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        if target is None:
            ax.scatter(*prediction.T, s=5, alpha=0.65, color="gray", label="partial input")
        else:
            ax.scatter(*target.T, s=5, alpha=0.45, color="royalblue", label="full target")
            ax.scatter(*prediction.T, s=5, alpha=0.55, color="red", label="model prediction")
        ax.scatter(0, 0, 0, marker="x", s=40, color="black", label="{} origin".format(frame_label))
        ax.set_title(title)
        ax.set_xlabel("{}-frame X (m)".format(frame_label))
        ax.set_ylabel("{}-frame Y (m)".format(frame_label))
        ax.set_zlabel("{}-frame Z (m)".format(frame_label))
        ax.legend(fontsize=8)
        points = prediction if target is None else np.concatenate((target, prediction), axis=0)
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
        radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 0.02)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=-60)
    fig.suptitle("RGB-D sample {} | joint two-stage PoinTr".format(sample_index))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print("Saved reconstruction figure: {}".format(output_path))
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/objects")
    parser.add_argument("--checkpoint", default="runs/rgbd_box_estimator/pointr_two_stage_best.pt")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.sample_index < 0:
        raise ValueError("--sample-index must be non-negative")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "pointr_rgbd_two_stage":
        raise ValueError(
            "Checkpoint must contain jointly trained stage1_model and stage2_model weights"
        )

    dataset_dir = Path(args.dataset)
    sample = _load_sample(dataset_dir, args.sample_index)
    num_points = int(checkpoint["num_points"])
    partial, complete_estimated, complete_object, origin, rotation = prepare_two_stage_clouds(
        sample["depth"], sample["object_mask"], sample["points_camera"],
        sample["pose_camera"], np.asarray(checkpoint["intrinsics"], dtype=np.float32), num_points,
        np.random.default_rng(args.sample_index), float(checkpoint.get("max_depth", 5.0)),
    )

    device = torch.device(args.device)
    stage1 = build_pointr_model(device)
    stage2 = build_pointr_model(device)
    stage1.load_state_dict(checkpoint["stage1_model"])
    stage2.load_state_dict(checkpoint["stage2_model"])
    stage1.eval()
    stage2.eval()
    with torch.no_grad():
        partial_tensor = torch.from_numpy(partial).unsqueeze(0).to(device)
        estimated_target_tensor = torch.from_numpy(complete_estimated).unsqueeze(0).to(device)
        object_target_tensor = torch.from_numpy(complete_object).unsqueeze(0).to(device)
        stage1_output = stage1(partial_tensor)[-1]
        stage1_chamfer = ChamferDistanceL1()(stage1_output, estimated_target_tensor).item()
        stage2_input = sample_points_fps(stage1_output, num_points)
        stage2_output = stage2(stage2_input)[-1][:, :stage2.num_pred]
        stage2_chamfer = ChamferDistanceL1()(stage2_output, object_target_tensor).item()
        stage1_prediction = sample_points_fps(stage1_output, num_points)[0].cpu().numpy()
        stage2_prediction = sample_points_fps(stage2_output, num_points)[0].cpu().numpy()
        registration_points = min(2048, stage1_output.shape[1], stage2_output.shape[1])
        stage1_registration = sample_points_fps(stage1_output, registration_points)[0].cpu().numpy()
        stage2_registration = sample_points_fps(stage2_output, registration_points)[0].cpu().numpy()

    predicted_rotation, predicted_translation, registration_residual = _icp_transform(
        stage1_registration, stage2_registration,
    )
    object_rotation_camera = Rotation.from_quat(sample["pose_camera"][3:]).as_matrix()
    true_rotation = rotation.T @ object_rotation_camera
    true_translation = (origin - sample["pose_camera"][:3]) @ object_rotation_camera
    frame_rotation_error = _rotation_error_degrees(predicted_rotation, true_rotation)
    frame_translation_error = float(np.linalg.norm(predicted_translation - true_translation))

    print("sample index: {}".format(args.sample_index))
    print("estimated input origin in camera CV (m): {}".format(np.array2string(origin, precision=5)))
    print("estimated input-frame rotation (columns are frame axes in camera CV):")
    print(np.array2string(rotation, precision=5))
    print("stage 1 target/prediction frame: estimated partial-cloud frame")
    print("stage 1 Chamfer L1: {:.6f} m".format(stage1_chamfer))
    print("stage 2 target/prediction frame: object asset root frame")
    print("stage 2 Chamfer L1: {:.6f} m".format(stage2_chamfer))
    print("estimated-frame to object-frame transform (row vectors: p_object = p_estimated @ R + t):")
    print("predicted R:\n{}".format(np.array2string(predicted_rotation, precision=5)))
    print("predicted t (m): {}".format(np.array2string(predicted_translation, precision=5)))
    print("ground-truth R:\n{}".format(np.array2string(true_rotation, precision=5)))
    print("ground-truth t (m): {}".format(np.array2string(true_translation, precision=5)))
    print("coordinate-frame rotation error: {:.2f} deg".format(frame_rotation_error))
    print("coordinate-frame translation error: {:.5f} m".format(frame_translation_error))
    print("ICP trimmed alignment residual: {:.5f} m".format(registration_residual))

    output_path = Path(args.output) if args.output else Path(
        "runs/rgbd_box_estimator/test_pointr_{:06d}.png".format(args.sample_index)
    )
    _show(
        partial, complete_estimated, stage1_prediction, complete_object,
        stage2_prediction, output_path, not args.no_display,
        stage1_chamfer, stage2_chamfer, args.sample_index,
    )


if __name__ == "__main__":
    main()
