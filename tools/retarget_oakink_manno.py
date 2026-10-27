#!/usr/bin/env python3
"""Decode OakInk MANO trajectories and retarget them to a right Inspire hand."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DEX_ROOT = ROOT / "external/dex-retargeting"


def _patch_legacy_numpy_aliases() -> None:
    """Allow old MANO/chumpy pickles to load with NumPy >= 1.24."""
    import builtins

    aliases = {"unicode": str}
    for name in ("bool", "int", "float", "complex", "object", "unicode", "str"):
        if name not in np.__dict__:
            setattr(np, name, aliases.get(name, getattr(builtins, name, str)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "datasets/oakink_human_trajectories.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/oakink_manno_inspire_retargeted.npz")
    parser.add_argument("--mano-models", type=Path, default=Path("/home/cyrus/ManipTrans/data/mano_v1_2/models"))
    parser.add_argument("--limit", type=int, default=None, help="Retarget only the first N sequences (for a quick check).")
    args = parser.parse_args()

    if not args.mano_models.is_dir():
        parser.error(f"MANO model directory not found: {args.mano_models}")
    sys.path.insert(0, str(DEX_ROOT / "src"))
    try:
        _patch_legacy_numpy_aliases()
        import smplx
        import torch
        from dex_retargeting.constants import HandType
        from dex_retargeting.retargeting_config import RetargetingConfig
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Install DexRetargeting dependencies "
            "(pin, nlopt, pytransform3d, anytree) in this Python environment."
        ) from exc

    data = np.load(args.input, allow_pickle=False)
    mano = np.asarray(data["mano_pose_coeffs"], dtype=np.float32)
    wrist = np.asarray(data["wrist_pose_world"], dtype=np.float32)
    count = len(mano) if args.limit is None else min(args.limit, len(mano))
    mano, wrist = mano[:count], wrist[:count]

    # smplx expects ``model_path/mano/MANO_RIGHT.pkl``; MANO downloads often
    # place the file directly in a ``models`` directory, so normalize both
    # layouts here.
    mano_root = args.mano_models
    if mano_root.name == "models" and not (mano_root / "mano").exists():
        # smplx hard-codes the final ``mano`` component in model_path.
        shim = Path(tempfile.mkdtemp(prefix="smplx-mano-"))
        (shim / "mano").symlink_to(mano_root, target_is_directory=True)
        mano_root = shim
    model = smplx.create(
        str(mano_root), model_type=" mano".strip(), is_rhand=True,
        use_pca=False, flat_hand_mean=False, batch_size=1, use_hands=True,
    )
    cfg_path = DEX_ROOT / "src/dex_retargeting/configs/offline/inspire_hand_right.yml"
    RetargetingConfig.set_default_urdf_dir(DEX_ROOT / "assets/robots/hands")
    config = RetargetingConfig.load_from_file(cfg_path)
    active_names = np.asarray(config.target_joint_names)
    retargeter = config.build()

    all_qpos = []
    for seq_idx in range(count):
        retargeter.reset()
        sequence_qpos = []
        for frame_idx in range(mano.shape[1]):
            # OakInk stores 16 local MANO quaternions in wxyz order. smplx
            # expects axis-angle root orientation followed by 15 joint poses.
            quat_wxyz = mano[seq_idx, frame_idx].reshape(16, 4)
            quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
            rotvec = Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
            pose = torch.from_numpy(rotvec.reshape(1, 48))
            with torch.no_grad():
                mano_output = model(
                    global_orient=pose[:, :3],
                    hand_pose=pose[:, 3:],
                    betas=torch.zeros((1, 10), dtype=pose.dtype),
                    transl=torch.zeros((1, 3), dtype=pose.dtype),
                )
            base_joints = mano_output.joints[0].cpu().numpy()
            vertices = mano_output.vertices[0].cpu().numpy()
            # smplx exposes the 16 MANO joints in kinematic order. Add the
            # five fingertip vertices and reorder to DexRetargeting's 21-point
            # MANO convention: wrist, thumb, index, middle, ring, little.
            tips = vertices[[745, 317, 444, 556, 673]]
            mano_xyz = np.concatenate(
                [base_joints[[0, 13, 14, 15]], tips[[0]],
                 base_joints[[1, 2, 3]], tips[[1]],
                 base_joints[[4, 5, 6]], tips[[2]],
                 base_joints[[10, 11, 12]], tips[[3]],
                 base_joints[[7, 8, 9]], tips[[4]]], axis=0
            ) / 1000.0
            # Convert MANO-local joints to OakInk world coordinates.
            root_r = Rotation.from_quat(wrist[seq_idx, frame_idx, 3:7]).as_matrix()
            root_p = wrist[seq_idx, frame_idx, :3]
            mano_xyz = mano_xyz @ root_r.T + root_p
            if frame_idx == 0:
                retargeter.warm_start(
                    root_p, quat_xyzw[0], hand_type=HandType.right,
                    is_mano_convention=True,
                )
            ref = mano_xyz[retargeter.optimizer.target_link_human_indices]
            sequence_qpos.append(retargeter.retarget(ref))
        all_qpos.append(np.stack(sequence_qpos))

    qpos = np.stack(all_qpos).astype(np.float32)
    joint_names = np.asarray(retargeter.joint_names)
    active_indices = np.asarray([list(joint_names).index(n) for n in active_names])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        wrist_pose_world=wrist,
        robot_qpos=qpos,
        robot_joint_names=joint_names,
        inspire_active_joint_position=qpos[..., active_indices],
        inspire_active_joint_names=active_names,
        source_path=data["source_path"][:count],
        object_id=data["object_id"][:count],
        primitive=data["primitive"][:count],
        wrist_pose_format=np.asarray("xyz+xyzw, world frame"),
        retargeting=np.asarray("DexRetargeting position / Inspire right hand; MANO beta=0"),
    )
    print(f"Saved {count} sequences x {mano.shape[1]} frames to {args.output}")
    print(f"robot_qpos: {qpos.shape}; active Inspire joints: {active_names.tolist()}")


if __name__ == "__main__":
    main()
