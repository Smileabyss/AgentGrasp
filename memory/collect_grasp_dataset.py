"""Collect successful grasp trajectories for the trajectory-memory module.

The Isaac Gym task and its checkpoint-backed one-step policy are used only to
produce successful samples. Hydra task overrides can be passed to this script.

Each saved sample contains:

* the object label and object-frame point cloud;
* the object's initial world-frame pose and its world-frame pose at each step;
* the Inspire palm-link pose trajectory (position + quaternion), stored under
  the legacy ``wrist_trajectory`` key for compatibility with the AE pipeline;
* the active palm/hand joint-position trajectory and the palm pose trajectory.

The trajectory state is sampled immediately before each environment action, so
its length is ``task.env.episodeLength``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig


def _parse_collector_args() -> argparse.Namespace:
    """Parse collector-only flags and leave Hydra's arguments untouched."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--output",
        default="datasets/grasp_dataset.npz",
        help="Output .npz file (default: datasets/grasp_dataset.npz).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of trajectories to save (default: 100 successful episodes).",
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        help="Save failed episodes too; success is still recorded per sample.",
    )
    parser.add_argument(
        "--no-trim-trajectories",
        action="store_true",
        help="Keep the complete episode instead of removing approach and post-lift frames.",
    )
    parser.add_argument(
        "--trim-hand-object-distance",
        type=float,
        default=0.35,
        help="Start saving once palm-to-object distance is at most this many metres (default: 0.35).",
    )
    parser.add_argument(
        "--trim-lift-height",
        type=float,
        default=0.1,
        help="Stop saving when object height has increased by this many metres (default: 0.1).",
    )
    parser.add_argument(
        "--trim-lift-steps",
        type=int,
        default=3,
        help="Require this many consecutive lifted frames before stopping (default: 3).",
    )
    parser.add_argument(
        "--trim-min-length",
        type=int,
        default=2,
        help="Minimum retained trajectory length (default: 2).",
    )
    parser.add_argument(
        "--no-symmetry-augmentation",
        action="store_true",
        help="Disable symmetry-based copies after trimming and resampling.",
    )
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args


COLLECTOR_ARGS = _parse_collector_args()
TARGET_TRAJECTORY_LENGTH = 20

AUTO_SYMMETRY_RULE = {
    "mode": "auto",
    "error_threshold": 0.02,
    # Point clouds are finite surface samples, so exact symmetric geometry
    # does not produce 100% point-to-point overlap after arbitrary angles.
    "overlap_threshold": 0.80,
    "overlap_tolerance_ratio": 0.035,
    "test_angles_degrees": [60.0, 120.0, 180.0],
    "max_order": 6,
    "augmentation_samples": 8,
    "max_points": 512,
    "max_variants": 8,
}


def _object_labels(env) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable string label and integer id for every vectorized env."""

    object_files = list(env.object_fns)
    labels = [str(object_files[i % len(object_files)]) for i in range(env.num_envs)]
    label_ids = [i % len(object_files) for i in range(env.num_envs)]
    return np.asarray(labels, dtype=np.str_), np.asarray(label_ids, dtype=np.int64)


def _stack_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        raise RuntimeError(
            "No episodes were collected. The policy produced no successful episode; "
            "rerun with --save-failures to inspect trajectories."
        )
    keys = records[0].keys()
    return {key: np.stack([record[key] for record in records], axis=0) for key in keys}


def _trim_trajectory(
    palm_pose: np.ndarray,
    object_pose: np.ndarray,
    hand_object_distance: float,
    lift_height: float,
    lift_steps: int,
    min_length: int,
) -> tuple[int, int]:
    """Return the inclusive/exclusive frame range containing the grasp motion.

    The approach starts at the first frame whose palm is close to the object.
    The post-grasp tail starts at the first sustained upward object displacement.
    If either event is absent, the corresponding episode boundary is retained.
    """
    if palm_pose.ndim != 2 or palm_pose.shape[-1] < 3:
        raise ValueError("palm_pose must have shape [T, >=3]")
    if object_pose.shape != (palm_pose.shape[0], 7):
        raise ValueError("object_pose must have shape [T, 7]")
    if hand_object_distance <= 0 or lift_height <= 0 or lift_steps <= 0 or min_length <= 0:
        raise ValueError("trim thresholds and min_length must be positive")

    distance = np.linalg.norm(palm_pose[:, :3] - object_pose[:, :3], axis=-1)
    close = np.flatnonzero(distance <= hand_object_distance)
    start = int(close[0]) if close.size else 0

    delta_z = object_pose[:, 2] - object_pose[0, 2]
    lifted = delta_z >= lift_height
    end = object_pose.shape[0]
    for index in range(0, object_pose.shape[0] - lift_steps + 1):
        if lifted[index : index + lift_steps].all():
            end = index
            break

    # Keep a usable prefix if noisy states trigger a threshold too early.
    end = max(end, start + min_length)
    end = min(end, object_pose.shape[0])
    if end <= start:
        start = max(0, end - min_length)
    return start, end


def _resample_trajectory(
    values: np.ndarray, target_length: int, normalize_quaternion: bool = False
) -> np.ndarray:
    """Uniformly resample a temporal array and optionally normalize quaternions."""
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("trajectory values must have shape [T, D] with T > 0")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    source_length = values.shape[0]
    source_time = np.linspace(0.0, 1.0, source_length)
    target_time = np.linspace(0.0, 1.0, target_length)
    result = np.stack(
        [np.interp(target_time, source_time, values[:, column]) for column in range(values.shape[1])],
        axis=-1,
    )
    if normalize_quaternion and values.shape[1] >= 7:
        quaternion = values[:, 3:7].copy()
        for index in range(1, source_length):
            if np.dot(quaternion[index - 1], quaternion[index]) < 0:
                quaternion[index] *= -1.0
        result[:, 3:7] = np.stack(
            [np.interp(target_time, source_time, quaternion[:, column]) for column in range(4)],
            axis=-1,
        )
        result[:, 3:7] /= np.linalg.norm(result[:, 3:7], axis=-1, keepdims=True).clip(min=1e-8)
    return result.astype(values.dtype, copy=False)


def _normalise_axis(axis: np.ndarray) -> Optional[np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    if axis.size != 3 or not np.isfinite(axis).all():
        return None
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return None
    axis = axis / norm
    # An axis and its negative describe the same rotations.  Canonicalising
    # the sign makes duplicate removal deterministic.
    first_nonzero = np.flatnonzero(np.abs(axis) > 1e-8)
    if first_nonzero.size and axis[first_nonzero[0]] < 0:
        axis = -axis
    return axis


def _deduplicate_axes(axes: list[np.ndarray], tolerance: float = 1e-4) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for axis in axes:
        normalised = _normalise_axis(axis)
        if normalised is None:
            continue
        if not any(abs(float(np.dot(normalised, other))) >= 1.0 - tolerance for other in result):
            result.append(normalised)
    return result


def _axis_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalise_axis(axis)
    if axis is None:
        raise ValueError("symmetry axis must be a non-zero finite 3-vector")
    x, y, z = axis
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return np.asarray(
        [
            [cosine + x * x * one_minus_cosine,
             x * y * one_minus_cosine - z * sine,
             x * z * one_minus_cosine + y * sine],
            [y * x * one_minus_cosine + z * sine,
             cosine + y * y * one_minus_cosine,
             y * z * one_minus_cosine - x * sine],
            [z * x * one_minus_cosine - y * sine,
             z * y * one_minus_cosine + x * sine,
             cosine + z * z * one_minus_cosine],
        ],
        dtype=np.float64,
    )


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to rotation matrices without scipy."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[-1] != 4:
        raise ValueError("quaternion must have shape [T, 4]")
    normalised = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1e-8)
    x, y, z, w = normalised.T
    return np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to xyzw quaternions."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape [T, 3, 3]")
    result = np.empty((matrix.shape[0], 4), dtype=np.float64)
    for index, current in enumerate(matrix):
        trace = float(np.trace(current))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(max(trace + 1.0, 1e-12))
            result[index] = [
                (current[2, 1] - current[1, 2]) / scale,
                (current[0, 2] - current[2, 0]) / scale,
                (current[1, 0] - current[0, 1]) / scale,
                0.25 * scale,
            ]
        else:
            diagonal = np.diag(current)
            largest = int(np.argmax(diagonal))
            if largest == 0:
                scale = 2.0 * np.sqrt(max(1.0 + current[0, 0] - current[1, 1] - current[2, 2], 1e-12))
                result[index] = [
                    0.25 * scale,
                    (current[0, 1] + current[1, 0]) / scale,
                    (current[0, 2] + current[2, 0]) / scale,
                    (current[2, 1] - current[1, 2]) / scale,
                ]
            elif largest == 1:
                scale = 2.0 * np.sqrt(max(1.0 - current[0, 0] + current[1, 1] - current[2, 2], 1e-12))
                result[index] = [
                    (current[0, 1] + current[1, 0]) / scale,
                    0.25 * scale,
                    (current[1, 2] + current[2, 1]) / scale,
                    (current[0, 2] - current[2, 0]) / scale,
                ]
            else:
                scale = 2.0 * np.sqrt(max(1.0 - current[0, 0] - current[1, 1] + current[2, 2], 1e-12))
                result[index] = [
                    (current[0, 2] + current[2, 0]) / scale,
                    (current[1, 2] + current[2, 1]) / scale,
                    0.25 * scale,
                    (current[1, 0] - current[0, 1]) / scale,
                ]
    result /= np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-8)
    return result


def _nearest_chamfer(points: np.ndarray, rotated: np.ndarray) -> float:
    """Return symmetric nearest-neighbour distance, normalised by object size."""
    try:
        from scipy.spatial import cKDTree

        first = cKDTree(points).query(rotated, workers=1)[0]
        second = cKDTree(rotated).query(points, workers=1)[0]
    except (ImportError, TypeError):
        def nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
            distances = []
            for start in range(0, len(query), 256):
                chunk = query[start : start + 256]
                squared = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=-1)
                distances.append(np.sqrt(np.min(squared, axis=1)))
            return np.concatenate(distances, axis=0)

        first = nearest(rotated, points)
        second = nearest(points, rotated)
    scale = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return float((first.mean() + second.mean()) * 0.5 / max(scale, 1e-8))


def _pointcloud_overlap(
    points: np.ndarray, rotated: np.ndarray, tolerance_ratio: float
) -> tuple[float, float, float]:
    """Measure the fraction of both point clouds that geometrically overlap.

    A point counts as overlapping when its nearest point in the other cloud is
    within ``tolerance_ratio * bbox_diagonal``.  The minimum of the two
    directional fractions is the conservative symmetric overlap score.
    """
    if tolerance_ratio <= 0:
        raise ValueError("overlap tolerance ratio must be positive")
    try:
        from scipy.spatial import cKDTree

        forward = cKDTree(points).query(rotated, workers=1)[0]
        backward = cKDTree(rotated).query(points, workers=1)[0]
    except (ImportError, TypeError):
        def nearest(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
            distances = []
            for start in range(0, len(query), 256):
                chunk = query[start : start + 256]
                squared = ((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(axis=-1)
                distances.append(np.sqrt(np.min(squared, axis=1)))
            return np.concatenate(distances, axis=0)

        forward = nearest(rotated, points)
        backward = nearest(points, rotated)
    tolerance = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) * tolerance_ratio
    forward_fraction = float(np.mean(forward <= tolerance))
    backward_fraction = float(np.mean(backward <= tolerance))
    return min(forward_fraction, backward_fraction), forward_fraction, backward_fraction


def _candidate_symmetry_axes(points: np.ndarray, rule: dict) -> list[np.ndarray]:
    configured = rule.get("axes", rule.get("candidate_axes", []))
    if configured is not None and len(configured) > 0:
        return _deduplicate_axes([np.asarray(axis, dtype=np.float64) for axis in configured])

    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered.T) if len(points) > 2 else np.eye(3)
    _, eigenvectors = np.linalg.eigh(covariance)
    # Prefer the object's z axis as the first stable representative, followed
    # by x and y. This is useful for nearly spherical tabletop objects where
    # no geometric axis is unique but the world/object vertical is meaningful.
    canonical = [
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
    ]
    candidates = canonical + [eigenvectors[:, index] for index in range(3)]
    # These diagonals cover common cube/tetrahedron-like symmetries while
    # keeping the candidate set small enough for collection-time detection.
    for first in range(3):
        for second in range(first + 1, 3):
            candidates.extend([canonical[first] + canonical[second], canonical[first] - canonical[second]])
    return _deduplicate_axes(candidates)


def _deduplicate_symmetry_detections(
    detections: list[dict], angle_tolerance_degrees: float = 5.0
) -> list[dict]:
    """Merge detections whose unoriented axes differ by only a few degrees."""
    cosine_threshold = np.cos(np.deg2rad(angle_tolerance_degrees))
    result: list[dict] = []
    for detection in detections:
        axis = _normalise_axis(detection["axis"])
        if axis is None:
            continue
        detection = dict(detection)
        detection["axis"] = axis
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(result)
                if abs(float(np.dot(axis, existing["axis"]))) >= cosine_threshold
            ),
            None,
        )
        if duplicate_index is None:
            result.append(detection)
            continue
        existing = result[duplicate_index]
        # Continuous detection is stronger evidence than a single cyclic
        # angle. Within the same type, retain the larger overlap and then the
        # smaller Chamfer diagnostic.
        existing_score = (
            bool(existing["continuous"]),
            float(existing.get("overlap", 0.0)),
            -float(existing.get("error", np.inf)),
        )
        candidate_score = (
            bool(detection["continuous"]),
            float(detection.get("overlap", 0.0)),
            -float(detection.get("error", np.inf)),
        )
        if candidate_score > existing_score:
            result[duplicate_index] = detection
    # Keep at most two representative axes. Additional detected directions
    # are usually PCA/candidate duplicates or unstable axes of near-spherical
    # clouds. The retained representatives must be approximately orthogonal.
    # Candidate generation is intentionally ordered as canonical object axes,
    # PCA axes, then diagonal axes. Preserve that order when selecting among
    # nearly indistinguishable scores so spherical clouds do not yield a
    # visually arbitrary PCA direction.
    orthogonal: list[dict] = []
    for detection in result:
        if len(orthogonal) >= 2:
            break
        if all(abs(float(np.dot(detection["axis"], kept["axis"]))) <= 0.2 for kept in orthogonal):
            orthogonal.append(detection)
    return orthogonal


def _prepare_symmetry_points(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("object point cloud must have shape [N, 3]")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return np.empty((0, 3), dtype=np.float64)
    if max_points > 0 and len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[indices]
    # Detection is translation invariant.  Augmentation itself remains about
    # the object-frame origin, which is the pose convention used by the task.
    return points - points.mean(axis=0, keepdims=True)


def _detect_symmetry_axes(points: np.ndarray, rule: dict) -> list[dict]:
    points = _prepare_symmetry_points(points, int(rule.get("max_points", 256)))
    if len(points) < 4:
        return []
    threshold = float(rule.get("error_threshold", rule.get("threshold", 0.03)))
    overlap_threshold = float(rule.get("overlap_threshold", 0.95))
    overlap_tolerance = float(rule.get("overlap_tolerance_ratio", 0.03))
    test_angles = [
        float(angle)
        for angle in rule.get("test_angles_degrees", [60.0, 120.0, 180.0])
        if 0.0 < float(angle) <= 180.0
    ]
    if not test_angles:
        raise ValueError("test_angles_degrees must contain positive angles")
    detections = []
    for axis in _candidate_symmetry_axes(points, rule):
        continuous_errors = []
        # Use large, explicit rotations. Small-angle agreement is not enough
        # to establish object-level rotational symmetry for a sparse cloud.
        for angle_degrees in test_angles:
            angle = np.deg2rad(angle_degrees)
            rotated = points @ _axis_rotation_matrix(axis, angle).T
            overlap, _, _ = _pointcloud_overlap(points, rotated, overlap_tolerance)
            continuous_errors.append((overlap, _nearest_chamfer(points, rotated)))
        if continuous_errors and min(item[0] for item in continuous_errors) >= overlap_threshold:
            detections.append(
                {
                    "axis": axis,
                    "order": 0,
                    "error": float(max(item[1] for item in continuous_errors)),
                    "overlap": float(min(item[0] for item in continuous_errors)),
                    "continuous": True,
                }
            )
    return _deduplicate_symmetry_detections(
        detections,
        float(rule.get("axis_merge_angle_degrees", 5.0)),
    )


def _symmetry_variants(points: np.ndarray, rule: dict) -> list[dict]:
    mode = str(rule.get("mode", "none")).lower()
    if mode in {"none", "off", "disabled"}:
        return []
    points = np.asarray(points, dtype=np.float64)
    symmetry_center = points.mean(axis=0)
    axes = _candidate_symmetry_axes(points, rule)
    detections = []
    if mode == "auto":
        detections = _detect_symmetry_axes(points, rule)
    elif mode in {"axial", "continuous", "spherical"}:
        samples = max(2, int(rule.get("samples", rule.get("augmentation_samples", 8))))
        for axis in axes:
            detections.append({"axis": axis, "order": 0, "error": 0.0, "continuous": True, "samples": samples})
    elif mode in {"discrete", "cyclic"}:
        order = int(rule.get("order", 2))
        if order < 2:
            raise ValueError("discrete symmetry order must be at least 2")
        for axis in axes:
            detections.append({"axis": axis, "order": order, "error": 0.0, "continuous": False})
    else:
        raise ValueError(f"unknown symmetry mode: {mode}")

    variants_by_axis: list[list[dict]] = []
    for detection in detections:
        axis = _normalise_axis(detection["axis"])
        if axis is None:
            continue
        if detection["continuous"]:
            count = max(2, int(detection.get("samples", rule.get("samples", rule.get("augmentation_samples", 8)))))
            angles = [2.0 * np.pi * index / count for index in range(1, count)]
        else:
            order = int(detection["order"])
            angles = [2.0 * np.pi * index / order for index in range(1, order)]
        axis_variants: list[dict] = []
        for angle in angles:
            matrix = _axis_rotation_matrix(axis, angle)
            if any(np.allclose(matrix, existing["matrix"], atol=1e-5) for existing in axis_variants):
                continue
            axis_variants.append(
                {
                    "matrix": matrix,
                    "center": symmetry_center.copy(),
                    "axis": axis.copy(),
                    "angle": float(angle),
                    "order": int(detection["order"]),
                    "error": float(detection["error"]),
                    "source": "auto" if mode == "auto" else "config",
                }
            )
        variants_by_axis.append(axis_variants)
    variants: list[dict] = []
    # Round-robin across axes so a two-axis object receives a total of N
    # samples, rather than N samples independently for each axis.
    max_variants = int(rule.get("max_variants", 0))
    while variants_by_axis and (max_variants <= 0 or len(variants) < max_variants):
        progressed = False
        for axis_variants in variants_by_axis:
            if axis_variants and (max_variants <= 0 or len(variants) < max_variants):
                variants.append(axis_variants.pop(0))
                progressed = True
        if not progressed:
            break
    return variants


def _transform_pose_by_object_symmetry(
    pose: np.ndarray,
    object_pose: np.ndarray,
    symmetry_matrix: np.ndarray,
    symmetry_center: np.ndarray,
) -> np.ndarray:
    object_position = object_pose[:, :3]
    object_rotation = _quaternion_to_matrix(object_pose[:, 3:7])
    palm_rotation = _quaternion_to_matrix(pose[:, 3:7])
    object_position_frame = np.einsum(
        "tji,tj->ti", object_rotation, pose[:, :3] - object_position
    )
    palm_rotation_frame = np.einsum("tji,tjk->tik", object_rotation, palm_rotation)

    transformed_position_frame = (
        (object_position_frame - symmetry_center) @ symmetry_matrix.T + symmetry_center
    )
    transformed_rotation_frame = np.einsum(
        "ij,tjk->tik", symmetry_matrix.T, palm_rotation_frame
    )
    transformed_position = object_position + np.einsum(
        "tj,tij->ti", transformed_position_frame, object_rotation
    )
    transformed_rotation = np.einsum(
        "tij,tjk->tik", object_rotation, transformed_rotation_frame
    )
    transformed = np.empty_like(pose, dtype=np.float64)
    transformed[:, :3] = transformed_position
    transformed[:, 3:7] = _matrix_to_quaternion(transformed_rotation)
    for index in range(1, len(transformed)):
        if np.dot(transformed[index - 1, 3:7], transformed[index, 3:7]) < 0:
            transformed[index, 3:7] *= -1.0
    return transformed.astype(pose.dtype, copy=False)


def _apply_symmetry_variant(record: dict[str, np.ndarray], variant: dict) -> dict[str, np.ndarray]:
    augmented = {key: np.array(value, copy=True) for key, value in record.items()}
    symmetry_matrix = variant["matrix"]
    symmetry_center = variant["center"]
    object_pose = np.asarray(record["object_pose_world_trajectory"])
    for key in ("wrist_trajectory", "palm_pose_trajectory"):
        augmented[key] = _transform_pose_by_object_symmetry(
            np.asarray(record[key]), object_pose, symmetry_matrix, symmetry_center
        )
    cloud = np.asarray(record["object_pointcloud"])
    augmented["object_pointcloud"] = (
        (cloud - symmetry_center) @ symmetry_matrix.T + symmetry_center
    )
    augmented.update(
        {
            "symmetry_augmented": np.asarray(True, dtype=np.bool_),
            "symmetry_axis": np.asarray(variant["axis"], dtype=np.float32),
            "symmetry_angle": np.asarray(variant["angle"], dtype=np.float32),
            "symmetry_source": np.asarray(variant["source"], dtype=np.str_),
            "symmetry_order": np.asarray(variant["order"], dtype=np.int64),
            "symmetry_error": np.asarray(variant["error"], dtype=np.float32),
        }
    )
    return augmented


def _set_identity_symmetry_metadata(record: dict[str, np.ndarray]) -> None:
    record.update(
        {
            "symmetry_augmented": np.asarray(False, dtype=np.bool_),
            "symmetry_axis": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            "symmetry_angle": np.asarray(0.0, dtype=np.float32),
            "symmetry_source": np.asarray("none", dtype=np.str_),
            "symmetry_order": np.asarray(1, dtype=np.int64),
            "symmetry_error": np.asarray(0.0, dtype=np.float32),
        }
    )


def _trim_record(
    record: dict[str, np.ndarray], args: argparse.Namespace, trim: bool = True
) -> None:
    """Trim a record and resample every temporal field to 20 frames."""
    if trim:
        start, end = _trim_trajectory(
            record["palm_pose_trajectory"],
            record["object_pose_world_trajectory"],
            args.trim_hand_object_distance,
            args.trim_lift_height,
            args.trim_lift_steps,
            args.trim_min_length,
        )
    else:
        start, end = 0, record["wrist_trajectory"].shape[0]
    record["trimmed_trajectory_length"] = np.asarray(end - start, dtype=np.int64)
    record["trajectory_length"] = np.asarray(TARGET_TRAJECTORY_LENGTH, dtype=np.int64)
    source_length = record["wrist_trajectory"].shape[0]
    for key, value in list(record.items()):
        if value.ndim < 2 or value.shape[0] != source_length:
            continue
        if key.endswith("_trajectory"):
            record[key] = _resample_trajectory(
                value[start:end],
                TARGET_TRAJECTORY_LENGTH,
                normalize_quaternion=key
                in {"wrist_trajectory", "palm_pose_trajectory", "object_pose_world_trajectory"},
            )


def collect_dataset(cfg: DictConfig, env, output_path: str, num_episodes: int, save_failures: bool) -> None:
    if num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if not cfg.test:
        raise ValueError("Dataset collection requires test=True and a trained checkpoint")
    if not cfg.checkpoint:
        raise ValueError("Dataset collection requires checkpoint=...")
    if not env.enable_pcl:
        raise ValueError("Enable object point clouds with task.env.enablePointCloud=True")

    # Load the saved actor directly for data collection; no optimizer or PPO
    # training loop is constructed.
    from memory.policy_inference import load_checkpoint_policy

    runner = load_checkpoint_policy(cfg, env)
    policy = runner.actor_critic
    policy.eval()

    labels, label_ids = _object_labels(env)
    symmetry_variants_by_object: dict[int, list[dict]] = {}
    if not COLLECTOR_ARGS.no_symmetry_augmentation:
        for object_id in range(len(env.object_fns)):
            variants = _symmetry_variants(env.object_pcls[object_id], AUTO_SYMMETRY_RULE)
            symmetry_variants_by_object[object_id] = variants
        variant_count = sum(len(variants) for variants in symmetry_variants_by_object.values())
        print(
            "Using point-cloud automatic symmetry detection; "
            f"detected {variant_count} non-identity variants across {len(env.object_fns)} objects"
        )
    else:
        print("Object symmetry augmentation disabled.")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    records: list[dict[str, np.ndarray]] = []
    episode_length = int(env.max_episode_length)

    while len(records) < num_episodes:
        reset_result = env.reset_idx(env_ids)
        current_obs = reset_result["obs"]
        current_states = env.get_state()
        with torch.no_grad():
            plan = policy(current_obs, current_states, inference=True)
        env.generate_reaching_plan_idx(env_ids, actions=plan)

        wrist_trajectory = torch.empty(
            (env.num_envs, episode_length, 7), device=env.device, dtype=torch.float32
        )
        palm_joint_trajectory = torch.empty(
            (env.num_envs, episode_length, env.num_active_hand_dofs),
            device=env.device,
            dtype=torch.float32,
        )
        palm_pose_trajectory = torch.empty(
            (env.num_envs, episode_length, 7), device=env.device, dtype=torch.float32
        )
        object_pose_trajectory = torch.empty(
            (env.num_envs, episode_length, 7), device=env.device, dtype=torch.float32
        )

        for timestep in range(episode_length):
            # Keep the legacy field name used by train_trajectory_vqvae.py,
            # but record the actual Inspire palm link rather than fr3_link8.
            wrist_trajectory[:, timestep, :3] = env.palm_pos
            wrist_trajectory[:, timestep, 3:] = env.palm_rot
            palm_joint_trajectory[:, timestep] = env.robot_dof_pos[
                :, env.active_hand_dof_indices
            ]
            palm_pose_trajectory[:, timestep, :3] = env.palm_pos
            palm_pose_trajectory[:, timestep, 3:] = env.palm_rot
            object_pose_trajectory[:, timestep] = env.object_pose
            env_action = env.compute_reference_actions()
            env.step(env_action)

        success = (env.successes > 0.5).detach().cpu().numpy().astype(bool)
        initial_object_pose = env.object_init_states[:, 0:7].detach().cpu().numpy()
        object_pointcloud = env.obj_pcl_buf.detach().cpu().numpy()
        wrist_np = wrist_trajectory.detach().cpu().numpy()
        palm_joint_np = palm_joint_trajectory.detach().cpu().numpy()
        palm_pose_np = palm_pose_trajectory.detach().cpu().numpy()
        object_pose_np = object_pose_trajectory.detach().cpu().numpy()

        candidate_ids = np.flatnonzero(np.ones_like(success, dtype=bool) if save_failures else success)
        for env_id in candidate_ids:
            if len(records) >= num_episodes:
                break
            record = {
                "object_label": labels[env_id],
                "object_label_id": label_ids[env_id],
                "object_pointcloud": object_pointcloud[env_id],
                "object_pose_world": initial_object_pose[env_id],
                "object_pose_world_trajectory": object_pose_np[env_id],
                "wrist_trajectory": wrist_np[env_id],
                "palm_joint_position_trajectory": palm_joint_np[env_id],
                "palm_pose_trajectory": palm_pose_np[env_id],
                "success": np.asarray(success[env_id], dtype=np.bool_),
            }
            _trim_record(
                record,
                COLLECTOR_ARGS,
                trim=not COLLECTOR_ARGS.no_trim_trajectories,
            )
            _set_identity_symmetry_metadata(record)
            variants = symmetry_variants_by_object.get(int(label_ids[env_id]), [])
            expanded_records = [record] + [
                _apply_symmetry_variant(record, variant) for variant in variants
            ]
            records.extend(expanded_records[: num_episodes - len(records)])
        print(
            f"Collected {len(records)}/{num_episodes} samples "
            f"({int(success.sum())}/{env.num_envs} successful this round)"
        )

    dataset = _stack_records(records)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)

    metadata = {
        "format": "DemoGrasp grasp trajectory dataset v4",
        "num_samples": int(len(records)),
        "episode_length": episode_length,
        "object_pointcloud_frame": "object",
        "object_pose_frame": "world",
        "wrist_pose_format": "xyz+xyzw",
        "wrist_trajectory_link": str(env.hand_specific_cfg["palm_link"]),
        "wrist_trajectory_semantics": "palm_link_pose",
        "palm_pose_format": "xyz+xyzw",
        "palm_joint_position_count": int(env.num_active_hand_dofs),
        "palm_joint_names": list(env.active_hand_dof_names),
        "successful_samples": int(dataset["success"].sum()),
        "trajectory_trimmed": not COLLECTOR_ARGS.no_trim_trajectories,
        "trim_hand_object_distance_m": COLLECTOR_ARGS.trim_hand_object_distance,
        "trim_lift_height_m": COLLECTOR_ARGS.trim_lift_height,
        "trim_lift_steps": COLLECTOR_ARGS.trim_lift_steps,
        "trim_min_length": COLLECTOR_ARGS.trim_min_length,
        "trajectory_target_length": TARGET_TRAJECTORY_LENGTH,
        "trimmed_trajectory_length_min": int(dataset["trimmed_trajectory_length"].min()),
        "trimmed_trajectory_length_max": int(dataset["trimmed_trajectory_length"].max()),
        "trimmed_trajectory_length_mean": float(dataset["trimmed_trajectory_length"].mean()),
        "trajectory_length_min": int(dataset["trajectory_length"].min()),
        "trajectory_length_max": int(dataset["trajectory_length"].max()),
        "trajectory_length_mean": float(dataset["trajectory_length"].mean()),
        "symmetry_augmentation": not COLLECTOR_ARGS.no_symmetry_augmentation,
        "symmetry_detection": "pointcloud_chamfer_auto",
        "symmetry_augmented_samples": int(dataset["symmetry_augmented"].sum()),
        "symmetry_variant_fraction": float(dataset["symmetry_augmented"].mean()),
        "checkpoint": os.path.abspath(str(cfg.checkpoint)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved dataset to {output} and metadata to {metadata_path}")


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig

    # This collector is intended to run without a viewer. Preserve an explicit
    # Hydra override such as ``headless=False`` when a viewer is needed.
    has_headless_override = any(
        override.split("=", 1)[0].lstrip("+") == "headless"
        for override in HydraConfig.get().overrides.task
    )
    if not has_headless_override:
        cfg.headless = True

    # Isaac Gym must load its CUDA bindings before PyTorch initializes.
    from isaacgym import gymapi  # noqa: F401
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import tasks  # Register the custom "grasp" task before env creation.

    # Isaac Gym must be imported before PyTorch.  Importing torch at module
    # scope makes gymdeps.py reject the process before the simulator starts.
    global torch
    import torch

    set_np_formatting()
    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        cfg.task.env.numEnvs,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        cfg.headless,
        cfg.multi_gpu,
        cfg.capture_video,
        cfg.force_render,
        cfg,
    )
    collect_dataset(
        cfg,
        env,
        COLLECTOR_ARGS.output,
        COLLECTOR_ARGS.num_episodes,
        COLLECTOR_ARGS.save_failures,
    )


if __name__ == "__main__":
    main()
