"""Collect full head-camera RGB-D frames, projected boxes, and simulator labels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/rgbd_object_full")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--bbox-jitter-px", type=float, default=8.0)
    parser.add_argument("--base-noise-xyz", type=float, nargs=3, default=(0.10, 0.10, 0.03))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args


ARGS = _parse_args()


def _quat_matrix(quat):
    return Rotation.from_quat(quat).as_matrix().astype(np.float32)


def _world_to_cv(points, camera_pose):
    camera_rotation = _quat_matrix(camera_pose[3:])
    points_isaac = (points - camera_pose[:3]) @ camera_rotation
    return np.stack((-points_isaac[:, 1], -points_isaac[:, 2], points_isaac[:, 0]), axis=-1)


def _object_pose_cv(object_pose, camera_pose):
    camera_rotation = Rotation.from_quat(camera_pose[3:]).inv()
    object_rotation = Rotation.from_quat(object_pose[3:])
    basis = Rotation.from_matrix(
        np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
    )
    rotation_cv = basis * camera_rotation * object_rotation
    translation_cv = _world_to_cv(object_pose[None, :3], camera_pose)[0]
    return np.concatenate((translation_cv, rotation_cv.as_quat().astype(np.float32)))


def _box_from_points(points_camera, intrinsics, width, height, jitter, rng):
    points = points_camera[points_camera[:, 2] > 1e-4]
    if len(points) < 4:
        return None
    u = intrinsics[0, 0] * points[:, 0] / points[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[:, 1] / points[:, 2] + intrinsics[1, 2]
    box = np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32)
    box += rng.uniform(-jitter, jitter, size=4).astype(np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return box


def _save_shard(output_dir, shard_index, samples):
    if not samples:
        return
    keys = samples[0].keys()
    arrays = {key: np.stack([sample[key] for sample in samples]) for key in keys}
    path = output_dir / f"shard_{shard_index:06d}.npz"
    np.savez_compressed(path, **arrays)
    print(f"Saved {len(samples)} samples to {path}")


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    output_dir = Path(ARGS.output)
    if list(output_dir.glob("shard_*.npz")):
        raise FileExistsError(
            "{} already contains dataset shards; choose a new empty output directory "
            "to avoid mixing old visible-only labels with full-object labels".format(output_dir)
        )

    # Isaac Gym must be imported before torch in this process.
    from isaacgym import gymapi, gymtorch
    import torch
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import tasks  # noqa: F401

    if cfg.task.hand_config.name != "g1_upper_right_wuji":
        raise ValueError("Collection is configured for hand=g1_upper_right_wuji")
    if ARGS.num_envs < 1 or ARGS.samples < 1:
        raise ValueError("--num-envs and --samples must be positive")

    rng = np.random.default_rng(ARGS.seed)
    cfg.seed = ARGS.seed
    cfg.num_envs = ARGS.num_envs
    cfg.task.env.numEnvs = ARGS.num_envs
    cfg.headless = True
    cfg.force_render = False
    cfg.task.env.render.enable = True
    cfg.task.env.render.camera_ids = [1]
    cfg.task.env.render.data_type = "rgb,depth,seg"
    cfg.task.env.render.appearance_realistic = True
    cfg.task.env.render.randomize = True
    cfg.task.env.render.randomization_params.camera_pos = [-0.02, 0.02]
    cfg.task.env.render.randomization_params.camera_quat = [-0.02, 0.02]
    cfg.task.env.render.randomization_params.table_xyz = [0.05, 0.02, 0.0]
    cfg.task.env.enableCameraSensors = True
    cfg.task.env.enablePointCloud = True
    cfg.task.env.asset.useDistractorObjects = False
    cfg.task.env.asset.randomizeObjectAssets = True
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.task.env.randomEpisodeLength = False
    cfg.task.env.resetRandomRot = "random"
    cfg.task.env.randomizeGraspPose = False

    camera = cfg.task.env.camera_config.camera_1
    camera.mount = "head_link"
    camera.width = ARGS.width
    camera.height = ARGS.height
    camera.depth_range = [0.1, 5.0]
    focal = ARGS.width / (2 * math.tan(math.radians(ARGS.fov_deg) / 2))
    camera.intrinsics = [[focal, 0, ARGS.width / 2], [0, focal, ARGS.height / 2], [0, 0, 1]]
    camera.position = [0.1, 0.0, 0.45]
    camera.quaternion = [0.0, math.sin(math.radians(22.5)), 0.0, math.cos(math.radians(22.5))]

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=False)
    env = isaacgymenvs.make(
        cfg.seed, cfg.task_name, ARGS.num_envs, cfg.sim_device, cfg.rl_device,
        cfg.graphics_device_id, True, cfg.multi_gpu, False, cfg.force_render, cfg,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env_ids = torch.arange(ARGS.num_envs, dtype=torch.long, device=env.device)
    rng = np.random.default_rng(ARGS.seed)
    samples, shard_index, collected = [], 0, 0
    intrinsics = np.asarray(env.camera_cfg["camera_1"]["intrinsics"], dtype=np.float32)
    camera_handles = env.camera_handles

    try:
        env.reset_idx(env_ids)
        base_xyz = env.root_state_tensor[env.robot_indices, :3].clone()
        while collected < ARGS.samples:
            env.reset_idx(env_ids)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            # Randomize the robot base in world XYZ; the attached head camera follows it.
            base_delta = rng.uniform(-1, 1, (ARGS.num_envs, 3)) * np.asarray(ARGS.base_noise_xyz)
            env.root_state_tensor[env.robot_indices, :3] = base_xyz + torch.as_tensor(
                base_delta, device=env.device, dtype=torch.float32)

            actor_ids = env.robot_indices.to(torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env.root_state_tensor),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids),
            )
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            try:
                rgb_batch = [tensor.cpu().numpy()[..., :3].copy() for tensor in env.camera_rgb_tensor_lists[0]]
                depth_batch = [tensor.cpu().numpy().copy() for tensor in env.camera_depth_tensor_lists[0]]
                seg_batch = [tensor.cpu().numpy().copy() for tensor in env.camera_seg_tensor_lists[0]]
            finally:
                env.gym.end_access_image_tensors(env.sim)

            env.gym.refresh_actor_root_state_tensor(env.sim)
            limit = min(ARGS.num_envs, ARGS.samples - collected)
            for i in range(limit):
                camera_transform = env.gym.get_camera_transform(env.sim, env.envs[i], camera_handles[i])
                camera_pose = np.array([
                    camera_transform.p.x, camera_transform.p.y, camera_transform.p.z,
                    camera_transform.r.x, camera_transform.r.y, camera_transform.r.z, camera_transform.r.w,
                ], dtype=np.float32)
                object_pose = env.root_state_tensor[env.object_indices[i], :7].cpu().numpy().copy()
                points_local = np.asarray(
                    env.object_pcls[env.object_asset_ids[i]], dtype=np.float32
                )
                points_world = points_local @ _quat_matrix(object_pose[3:]).T + object_pose[:3]
                points_camera = _world_to_cv(points_world, camera_pose)
                box = _box_from_points(points_camera, intrinsics, ARGS.width, ARGS.height, ARGS.bbox_jitter_px, rng)
                if box is None:
                    continue
                if not np.any(seg_batch[i] == 2):
                    continue
                point_count = 512
                if len(points_camera) >= point_count:
                    selected = rng.choice(len(points_camera), point_count, replace=False)
                else:
                    selected = rng.choice(len(points_camera), point_count, replace=True)
                pose_camera = _object_pose_cv(object_pose, camera_pose)
                samples.append({
                    "rgb": rgb_batch[i],
                    "depth": depth_batch[i],  # raw negative Isaac Gym IMAGE_DEPTH values
                    "bbox": box,
                    "object_mask": (seg_batch[i] == 2).astype(np.uint8),
                    "object_asset_id": np.int32(env.object_asset_ids[i]),
                    "points_camera": points_camera[selected].astype(np.float32),
                    "pose_camera": pose_camera.astype(np.float32),
                })
                collected += 1
                if len(samples) >= ARGS.shard_size:
                    _save_shard(output_dir, shard_index, samples)
                    shard_index += 1
                    samples.clear()
            print(f"Collected {collected}/{ARGS.samples}")

        _save_shard(output_dir, shard_index, samples)
        metadata = {
            "width": ARGS.width, "height": ARGS.height, "intrinsics": intrinsics.tolist(),
            "depth_convention": "raw Isaac Gym IMAGE_DEPTH; negative meters",
            "point_frame": "camera CV: right, down, forward; meters",
            "pose_frame": "camera CV; translation meters, quaternion xyzw",
            "point_cloud_label_type": "full_asset_surface",
            "object_urdf_files": list(env.object_fns),
            "point_cloud_label": (
                "512 points from the complete target asset point cloud transformed to camera "
                "CV coordinates, including surfaces occluded in the RGB-D frame"
            ),
            "bbox_order": "xmin, ymin, xmax, ymax pixels",
            "camera_position_head_link": [0.1, 0.0, 0.45],
            "camera_rpy_deg_head_link": [0.0, 45.0, 0.0],
            "samples": collected,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
