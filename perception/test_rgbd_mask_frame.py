"""Estimate and visualize an object frame from a masked depth point cloud."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from perception.rgbd_frame import estimate_frame as _estimate_frame
from perception.rgbd_frame import masked_depth_to_points as _masked_depth_to_points


def _load_sample(dataset_dir, global_index):
    shard_paths = sorted(Path(dataset_dir).glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError("No shard_*.npz files found in {}".format(dataset_dir))
    offset = 0
    for path in shard_paths:
        with np.load(path) as shard:
            count = len(shard["pose_camera"])
            if global_index < offset + count:
                index = global_index - offset
                return {
                    key: shard[key][index].copy()
                    for key in ("rgb", "depth", "object_mask", "pose_camera")
                }
        offset += count
    raise IndexError("sample index {} is outside [0, {})".format(global_index, offset))


def _draw_frame(ax, origin, rotation, length, linestyle, label):
    for i, color in enumerate(("red", "green", "blue")):
        direction = rotation[:, i] * length
        ax.quiver(
            *origin, *direction, color=color, arrow_length_ratio=0.18,
            linewidth=2, linestyle=linestyle,
        )
        endpoint = origin + direction
        ax.text(*endpoint, "XYZ"[i], color=color, fontsize=9, weight="bold")
    ax.scatter(*origin, marker="o", s=42, label=label)


def _rotation_error_deg(estimated, target):
    relative = target.T @ estimated
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _show(sample, points, estimated_origin, estimated_rotation, output_path, show):
    mask = sample["object_mask"].astype(bool)
    target_pose = sample["pose_camera"].astype(np.float32)
    target_origin = target_pose[:3]
    target_rotation = Rotation.from_quat(target_pose[3:]).as_matrix().astype(np.float32)
    origin_error = float(np.linalg.norm(estimated_origin - target_origin))
    rotation_error = _rotation_error_deg(estimated_rotation, target_rotation)

    rgb = sample["rgb"].astype(np.float32) / 255.0
    overlay = rgb.copy()
    overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([0.0, 1.0, 0.0])
    depth = np.where(np.isfinite(sample["depth"]) & (sample["depth"] < 0),
                     -sample["depth"], np.nan)

    fig = plt.figure(figsize=(14, 9))
    ax_rgb = fig.add_subplot(2, 2, 1)
    ax_rgb.imshow(overlay)
    ax_rgb.set_title("RGB with object mask")
    ax_rgb.axis("off")

    ax_depth = fig.add_subplot(2, 2, 2)
    valid_depth = depth[np.isfinite(depth)]
    lo, hi = np.percentile(valid_depth, (2, 98))
    depth_plot = ax_depth.imshow(depth, cmap="magma", vmin=lo, vmax=max(hi, lo + 1e-6))
    ax_depth.set_title("Depth (meters)")
    ax_depth.axis("off")
    fig.colorbar(depth_plot, ax=ax_depth, fraction=0.046, pad=0.04, label="meters")

    ax = fig.add_subplot(2, 1, 2, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.35,
               color="gray", label="mask-filtered depth cloud")
    span = max(float(np.ptp(points, axis=0).max()), 0.02)
    axis_length = span * 0.35
    _draw_frame(ax, target_origin, target_rotation, axis_length, "solid", "true origin")
    _draw_frame(ax, estimated_origin, estimated_rotation, axis_length, "dashed", "estimated origin")
    ax.set_title(
        "True frame (solid) and estimated frame (dashed)\n"
        "origin error {:.3f} m · frame rotation error {:.1f}°".format(
            origin_error, rotation_error,
        )
    )
    ax.set_xlabel("X right (m)")
    ax.set_ylabel("Y down (m)")
    ax.set_zlabel("Z forward (m)")
    ax.legend(fontsize=8)
    combined = np.concatenate((points, target_origin[None], estimated_origin[None]), axis=0)
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2.0
    radius = max(float(np.ptp(combined, axis=0).max()) / 2.0, 0.02)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    fig.suptitle("Red=X, green=Y, blue=Z; estimated axes are dashed", fontsize=10)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print("Saved comparison figure: {}".format(output_path))
    print("masked depth points: {}".format(len(points)))
    print("estimated origin (camera CV, m): {}".format(
        np.array2string(estimated_origin, precision=5)))
    print("true origin (camera CV, m): {}".format(
        np.array2string(target_origin, precision=5)))
    print("origin error: {:.5f} m".format(origin_error))
    print("frame rotation error: {:.2f} deg".format(rotation_error))
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/rgbd_object")
    parser.add_argument("--sample-index", type=int, default=0,
                        help="global sample index across sorted shard_*.npz files")
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    if args.sample_index < 0:
        parser.error("--sample-index must be non-negative")
    if args.max_depth <= 0:
        parser.error("--max-depth must be positive")

    dataset_dir = Path(args.dataset)
    metadata = json.loads((dataset_dir / "metadata.json").read_text())
    sample = _load_sample(dataset_dir, args.sample_index)
    points = _masked_depth_to_points(
        sample["depth"], sample["object_mask"],
        np.asarray(metadata["intrinsics"], dtype=np.float32), args.max_depth,
    )
    estimated_origin, estimated_rotation = _estimate_frame(points)
    output_path = Path(args.output) if args.output else Path(
        "runs/rgbd_box_estimator/test_mask_frame_{:06d}.png".format(args.sample_index)
    )
    _show(sample, points, estimated_origin, estimated_rotation,
          output_path, not args.no_display)


if __name__ == "__main__":
    main()
