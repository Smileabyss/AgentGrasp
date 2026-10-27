"""Geometry helpers for RGB-D point clouds and estimated object frames."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def masked_depth_to_points(depth, mask, intrinsics, max_depth):
    """Back-project all valid masked negative Isaac Gym depths into camera-CV xyz."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(depth)
        & (depth < -1e-6)
        & (-depth <= max_depth)
    )
    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError("object mask contains no valid depth pixels")
    z = -depth[v, u]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if fx <= 0 or fy <= 0:
        raise ValueError("camera focal lengths fx and fy must be positive")
    points = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
    return points.astype(np.float32)


def estimate_frame(points):
    """Use the visible-cloud centroid, image-up z, and horizontal PCA x."""
    origin = points.mean(axis=0)
    z_axis = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    horizontal = points.astype(np.float64) - origin
    horizontal -= np.outer(horizontal @ z_axis, z_axis)
    covariance = horizontal.T @ horizontal / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    x_axis = eigenvectors[:, np.argmax(eigenvalues)]
    x_axis -= np.dot(x_axis, z_axis) * z_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= x_norm
    if x_axis[0] < 0:
        x_axis = -x_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return origin.astype(np.float32), rotation.astype(np.float32)


def prepare_estimated_clouds(
    depth, mask, complete_points_camera, intrinsics, num_partial_points,
    rng, max_depth=5.0,
):
    """Sample and transform partial and complete clouds into the estimated frame."""
    partial_camera = masked_depth_to_points(depth, mask, intrinsics, max_depth)
    origin, rotation = estimate_frame(partial_camera)
    selected = rng.choice(
        len(partial_camera), size=num_partial_points,
        replace=len(partial_camera) < num_partial_points,
    )
    partial = (partial_camera[selected] - origin) @ rotation
    complete = (np.asarray(complete_points_camera, dtype=np.float32) - origin) @ rotation
    complete_indices = rng.choice(
        len(complete), size=num_partial_points,
        replace=len(complete) < num_partial_points,
    )
    return partial.astype(np.float32), complete[complete_indices].astype(np.float32), origin, rotation


def prepare_two_stage_clouds(
    depth, mask, complete_points_camera, object_pose_camera, intrinsics,
    num_points, rng, max_depth=5.0,
):
    """Prepare partial, estimated-frame complete, and object-frame target clouds."""
    partial_camera = masked_depth_to_points(depth, mask, intrinsics, max_depth)
    origin, rotation = estimate_frame(partial_camera)
    partial_indices = rng.choice(
        len(partial_camera), size=num_points,
        replace=len(partial_camera) < num_points,
    )
    partial = (partial_camera[partial_indices] - origin) @ rotation

    complete_camera = np.asarray(complete_points_camera, dtype=np.float32)
    complete_indices = rng.choice(
        len(complete_camera), size=num_points,
        replace=len(complete_camera) < num_points,
    )
    complete_camera = complete_camera[complete_indices]
    complete_estimated = (complete_camera - origin) @ rotation

    pose = np.asarray(object_pose_camera, dtype=np.float32)
    object_rotation_camera = Rotation.from_quat(pose[3:]).as_matrix().astype(np.float32)
    complete_local = (complete_camera - pose[:3]) @ object_rotation_camera
    return (
        partial.astype(np.float32),
        complete_estimated.astype(np.float32),
        complete_local.astype(np.float32),
        origin,
        rotation,
    )
