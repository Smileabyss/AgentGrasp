#!/usr/bin/env python3
"""Convert OakInk-V2 annotations into compact object-relative trajectories.

This reads only ``anno_preview/*.pkl``. Images, depth maps, videos, MANO
meshes, and Isaac Gym are not needed. Each output sample is one
sequence/object pair resampled to a fixed number of frames.

The exported ``human_trajectory`` is 9-D per frame:
object-frame wrist position (3) and wrist rotation 6-D (6). MANO pose
coefficients are kept separately because they cannot be silently converted to
the six Inspire joint values expected by the current robot AE.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _array(value: Any) -> np.ndarray:
    """Convert numpy/torch scalar containers without requiring torch."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _quat_to_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Stable enough conversion for trajectory export (no scipy dependency)."""
    r = np.asarray(matrix, dtype=np.float64)
    out = np.empty(r.shape[:-2] + (4,), dtype=np.float64)
    trace = np.trace(r, axis1=-2, axis2=-1)
    positive = trace > 0
    s = np.sqrt(np.maximum(trace[positive] + 1.0, 1e-12)) * 2
    out[positive, 0] = 0.25 * s
    out[positive, 1] = (r[positive, 2, 1] - r[positive, 1, 2]) / s
    out[positive, 2] = (r[positive, 0, 2] - r[positive, 2, 0]) / s
    out[positive, 3] = (r[positive, 1, 0] - r[positive, 0, 1]) / s
    remaining = ~positive
    # A vectorized branch for the largest diagonal is unnecessary here; the
    # annotation files are small after resampling, so use a clear loop.
    for index in zip(*np.where(remaining)):
        m = r[index]
        diagonal = np.diag(m)
        k = int(np.argmax(diagonal))
        if k == 0:
            s = 2 * np.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
            out[index] = [(m[2, 1] - m[1, 2]) / s, s / 4,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif k == 1:
            s = 2 * np.sqrt(max(1 - m[0, 0] + m[1, 1] - m[2, 2], 1e-12))
            out[index] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          s / 4, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(max(1 - m[0, 0] - m[1, 1] + m[2, 2], 1e-12))
            out[index] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, s / 4]
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-8)


def _resample(values: np.ndarray, steps: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, steps)
    flat = values.reshape(values.shape[0], -1)
    result = np.stack([np.interp(target, source, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
    return result.reshape((steps,) + values.shape[1:])


def _resample_rotation(rotations: np.ndarray, steps: int) -> np.ndarray:
    if rotations.shape[0] == 1:
        return np.repeat(rotations, steps, axis=0)
    source_times = np.linspace(0.0, 1.0, rotations.shape[0])
    target_times = np.linspace(0.0, 1.0, steps)
    return Slerp(source_times, Rotation.from_matrix(rotations))(target_times).as_matrix()


def _resample_mano_quaternions(values: np.ndarray, steps: int) -> np.ndarray:
    """Resample OakInk's 16 quaternion MANO pose coefficients by SLERP."""
    values = np.asarray(values)
    values = values.reshape(values.shape[0], -1, 4).astype(np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
    source_position = np.linspace(0.0, values.shape[0] - 1, steps)
    lower = np.floor(source_position).astype(np.int64)
    upper = np.minimum(lower + 1, values.shape[0] - 1)
    fraction = source_position - lower
    q0 = values[lower]
    q1 = values[upper]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    alpha = fraction[:, None, None]
    linear = sin_theta < 1e-6
    w0 = np.where(linear, 1.0 - alpha, np.sin((1.0 - alpha) * theta) / np.maximum(sin_theta, 1e-8))
    w1 = np.where(linear, alpha, np.sin(alpha * theta) / np.maximum(sin_theta, 1e-8))
    result = w0 * q0 + w1 * q1
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8)
    return result.reshape(steps, -1)


def _load_hand_frame(frame: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    translation = _array(frame["rh__tsl"]).reshape(-1, 3)[0].astype(np.float64)
    pose = _array(frame["rh__pose_coeffs"]).reshape(-1, 4)
    if pose.shape[0] < 1:
        raise ValueError("MANO frame has no pose coefficients")
    # OakInk stores the root quaternion in the first pose coefficient. The
    # remaining 15 coefficients are retained verbatim in mano_pose_coeffs.
    return translation, pose.astype(np.float64)


def convert(source: Path, output: Path, steps: int, max_sequences: int | None, program_dir: Path) -> None:
    records: List[np.ndarray] = []
    mano_records: List[np.ndarray] = []
    object_records: List[np.ndarray] = []
    wrist_world_records: List[np.ndarray] = []
    source_names: List[str] = []
    object_names: List[str] = []
    primitive_names: List[str] = []
    source_ranges: List[np.ndarray] = []
    files = sorted(source.glob("*.pkl"))
    if max_sequences is not None:
        files = files[:max_sequences]
    for file_index, file_path in enumerate(files, start=1):
        with file_path.open("rb") as handle:
            annotation = pickle.load(handle)
        frame_ids = list(annotation["mocap_frame_id_list"])
        raw_mano = annotation["raw_mano"]
        program_path = program_dir / f"{file_path.stem}.json"
        if not program_path.exists():
            continue
        program = json.loads(program_path.read_text())
        for stage_key, stage in program.items():
            try:
                left_range, right_range = ast.literal_eval(stage_key)
            except (ValueError, SyntaxError):
                continue
            if right_range is None:
                continue
            if left_range is None:
                begin, end = right_range
            else:
                begin, end = max(left_range[0], right_range[0]), min(left_range[1], right_range[1])
                if begin >= end:
                    continue
            for object_id in stage.get("obj_list_rh") or []:
                transforms = annotation["obj_transf"].get(object_id)
                if transforms is None:
                    continue
                valid = [f for f in frame_ids if begin <= f <= end and f in raw_mano and f in transforms]
                if len(valid) < 2:
                    continue
                hand_pos, hand_quat, mano_pose, object_pose = [], [], [], []
                for frame_id in valid:
                    pos, pose = _load_hand_frame(raw_mano[frame_id])
                    hand_pos.append(pos)
                    hand_quat.append(pose[0])
                    mano_pose.append(pose.reshape(-1))
                    object_pose.append(_array(transforms[frame_id]).reshape(4, 4))
                hand_pos = _resample(np.asarray(hand_pos), steps)
                hand_rot_world = _resample_rotation(_quat_to_matrix_wxyz(np.asarray(hand_quat)), steps)
                mano_pose = _resample_mano_quaternions(mano_pose, steps)
                raw_object_pose = np.asarray(object_pose)
                object_rot = _resample_rotation(raw_object_pose[:, :3, :3], steps)
                object_pos = _resample(raw_object_pose[:, :3, 3], steps)
                object_pose = np.tile(np.eye(4, dtype=np.float64), (steps, 1, 1))
                object_pose[:, :3, :3] = object_rot
                object_pose[:, :3, 3] = object_pos
                wrist_pos_obj = np.einsum("tij,tj->ti", object_rot.transpose(0, 2, 1), hand_pos - object_pos)
                wrist_rot_obj = np.einsum("tij,tjk->tik", object_rot.transpose(0, 2, 1), hand_rot_world)
                rotation6d = wrist_rot_obj[:, :, :2].transpose(0, 2, 1).reshape(steps, 6)
                records.append(np.concatenate((wrist_pos_obj, rotation6d), axis=-1).astype(np.float32))
                mano_records.append(mano_pose.astype(np.float32))
                object_records.append(object_pose.astype(np.float32))
                wrist_world_quat = _matrix_to_quat_wxyz(hand_rot_world)
                wrist_world_records.append(np.concatenate((hand_pos, wrist_world_quat), axis=-1).astype(np.float32))
                source_names.append(str(file_path))
                object_names.append(str(object_id))
                primitive_names.append(str(stage.get("primitive_rh", "unknown")))
                source_ranges.append(np.asarray([valid[0], valid[-1], len(valid)], dtype=np.int64))
        if file_index % 10 == 0 or file_index == len(files):
            print(f"processed {file_index}/{len(files)} annotation files; exported {len(records)} trajectories", flush=True)
    if not records:
        raise RuntimeError(f"No valid OakInk sequences found under {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        human_trajectory=np.stack(records),
        mano_pose_coeffs=np.stack(mano_records),
        object_pose_world=np.stack(object_records),
        wrist_pose_world=np.stack(wrist_world_records),
        source_path=np.asarray(source_names),
        object_id=np.asarray(object_names),
        primitive=np.asarray(primitive_names),
        source_frame_range_count=np.stack(source_ranges),
        representation=np.asarray("object-relative wrist xyz + rotation6d; MANO pose retained separately"),
    )
    print(f"saved {len(records)} sequence/object trajectories to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/home/cyrus/ManipTrans/data/OakInk-v2/anno_preview"))
    parser.add_argument("--output", type=Path, default=Path("datasets/oakink_human_trajectories.npz"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--program-dir", type=Path, default=Path("/home/cyrus/ManipTrans/data/OakInk-v2/program/program_info"))
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    convert(args.source, args.output, args.steps, args.max_sequences, args.program_dir)


if __name__ == "__main__":
    main()
