#!/usr/bin/env python3
"""Expand the dataset's six Inspire hand channels to Wuji's 20 joints.

This follows the approximate joint-sharing map used by the project at replay
time, with a curl gain so the independently actuated Wuji finger joints bend
visibly. It is not geometric IK.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "datasets/inspire_grasp.npz"
DEFAULT_OUTPUT = ROOT / "datasets/wuji_grasp.npz"
HAND_CONFIG = ROOT / "tasks/hand/g1_upper_right_wuji.yaml"
DEFAULT_CURL_GAIN = 2.0


def _target_source_names(target_joint_names: list[str]) -> list[str]:
    """Map Wuji's thumb/index/middle/ring/little joint order by semantics."""
    result = []
    finger_source = {
        2: "right_index_1_joint",
        3: "right_middle_1_joint",
        4: "right_ring_1_joint",
        5: "right_little_1_joint",
    }
    for target_name in target_joint_names:
        if target_name.startswith("right_finger1_joint"):
            source_name = (
                "right_thumb_1_joint"
                if target_name.endswith("joint1")
                else "right_thumb_2_joint"
            )
        else:
            finger = int(target_name[len("right_finger"):].split("_", 1)[0])
            if finger not in finger_source:
                raise ValueError(f"Unexpected Wuji joint name: {target_name}")
            source_name = finger_source[finger]
        result.append(source_name)
    return result


def _joint_limits(urdf_path: Path, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise ValueError(f"URDF has no position limits for: {missing}")
    values = np.asarray([limits[name] for name in joint_names], dtype=np.float32)
    return values[:, 0], values[:, 1]


def convert(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
    curl_gain: float = DEFAULT_CURL_GAIN,
) -> None:
    metadata_path = input_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it")

    metadata = json.loads(metadata_path.read_text())
    source_names = metadata.get("palm_joint_names")
    if not isinstance(source_names, list):
        raise ValueError("Dataset metadata must contain palm_joint_names")

    with np.load(input_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    source = np.asarray(arrays["palm_joint_position_trajectory"], dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != len(source_names):
        raise ValueError(
            "palm_joint_position_trajectory must have shape [N, T, len(palm_joint_names)]"
        )
    if not np.isfinite(source).all():
        raise ValueError("Source hand trajectory contains non-finite values")

    config = yaml.safe_load(HAND_CONFIG.read_text())
    target_names = list(config["active_hand_dof_names"])
    source_index = {name: index for index, name in enumerate(source_names)}
    source_for_target = _target_source_names(target_names)
    missing = sorted(set(source_for_target) - source_index.keys())
    if missing:
        raise ValueError(f"Dataset is missing Inspire joints required for mapping: {missing}")

    urdf_path = ROOT / "assets" / config["robotAssetFile"]
    lower, upper = _joint_limits(urdf_path, target_names)
    mapped = np.stack([source[..., source_index[name]] for name in source_for_target], axis=-1)
    curled = mapped * curl_gain
    clipped = np.clip(curled, lower, upper).astype(np.float32, copy=False)

    arrays["inspire_palm_joint_position_trajectory"] = source
    arrays["palm_joint_position_trajectory"] = clipped
    arrays["palm_joint_names"] = np.asarray(target_names)
    arrays["hand_retargeting_method"] = np.asarray(
        f"name-based Inspire-to-Wuji joint sharing with curl gain {curl_gain:g}, "
        "clipped to Wuji URDF limits"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)

    output_metadata = dict(metadata)
    output_metadata["palm_joint_position_count"] = len(target_names)
    output_metadata["palm_joint_names"] = target_names
    output_metadata["hand_retargeting"] = {
        "method": "joint-name sharing with curl gain and Wuji URDF-limit clipping",
        "curl_gain": float(curl_gain),
        "source_metadata": str(metadata_path),
        "source_joint_names": source_names,
        "target_joint_names": target_names,
        "source_joint_for_target": source_for_target,
        "geometric_ik": False,
        "note": "Wrist/palm poses are copied from the source dataset; only hand joint channels are mapped.",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2) + "\n"
    )

    clamp_fraction = float(np.not_equal(curled, clipped).mean())
    print(f"Saved {clipped.shape[0]} trajectories x {clipped.shape[1]} frames x {clipped.shape[2]} Wuji joints")
    print(f"NPZ: {output_path}")
    print(f"Metadata: {output_path.with_suffix('.json')}")
    print(f"Values clipped to target URDF limits: {clamp_fraction:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--curl-gain",
        type=float,
        default=DEFAULT_CURL_GAIN,
        help="Scale Inspire finger angles before applying Wuji joint limits (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.curl_gain <= 0:
        parser.error("--curl-gain must be positive")
    convert(args.input, args.output, args.overwrite, args.curl_gain)


if __name__ == "__main__":
    main()
