"""Collect balanced RGB-D observations and complete object point-cloud labels."""

from __future__ import annotations

print("Starting RGB-D object collector; loading Python dependencies...", flush=True)

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/object_partial_points")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--samples", type=int, default=None, help="total samples, split evenly across objects")
    parser.add_argument("--samples-per-object", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--bbox-jitter-px", type=float, default=8.0)
    parser.add_argument("--base-noise-xyz", type=float, nargs=3, default=(0.10, 0.10, 0.03))
    parser.add_argument("--max-attempts-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_asset-offset", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_object-list", default=None, help=argparse.SUPPRESS)
    args, hydra_argv = parser.parse_known_args()
    args.hydra_argv = hydra_argv
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
    path = output_dir / "shard_{:06d}.npz".format(shard_index)
    np.savez_compressed(path, **arrays)
    print("Saved {} samples to {}".format(len(samples), path))


def _run_group_workers(
    cfg, asset_root, asset_names, original_list, output_dir,
    samples_per_object, yaml,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metadata.json").exists():
        raise FileExistsError("{} already contains a completed dataset".format(output_dir))

    existing_counts = np.zeros(len(asset_names), dtype=np.int64)
    existing_shards = sorted(output_dir.glob("shard_*.npz"))
    for path in existing_shards:
        with np.load(path, allow_pickle=False) as shard:
            asset_ids = np.asarray(shard["object_asset_id"], dtype=np.int64)
        if np.any((asset_ids < 0) | (asset_ids >= len(asset_names))):
            raise ValueError("{} contains object_asset_id values outside the configured object list".format(path))
        np.add.at(existing_counts, asset_ids, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    group_size = min(ARGS.num_envs, len(asset_names))
    group_count = (len(asset_names) + group_size - 1) // group_size
    print(
        "Dataset plan: {} objects, {} samples/object ({} total), {} envs per group, {} groups.".format(
            len(asset_names), samples_per_object, len(asset_names) * samples_per_object,
            group_size, group_count,
        ),
        flush=True,
    )
    if existing_shards:
        print(
            "Resuming from {} existing samples.".format(int(existing_counts.sum())),
            flush=True,
        )

    next_shard = max(
        (int(path.stem.split("_")[-1]) for path in existing_shards),
        default=-1,
    ) + 1
    total_collected = int(existing_counts.sum())
    asset_dir = (Path(asset_root).resolve() / Path(original_list)).parent
    for group_number, start in enumerate(range(0, len(asset_names), group_size), start=1):
        end = min(start + group_size, len(asset_names))
        names = asset_names[start:end]
        group_counts = existing_counts[start:end]
        if np.all(group_counts == samples_per_object):
            print("Skipping complete object group {}/{}.".format(group_number, group_count), flush=True)
            continue
        if np.any(group_counts != 0):
            raise ValueError(
                "Object group {}/{} has partial or excess samples. Resume only works when a group is "
                "complete or untouched; use a new output directory to recollect it.".format(
                    group_number, group_count
                )
            )

        print("Launching isolated worker for object group {}/{}...".format(group_number, group_count), flush=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=".collect_object_partical_",
            dir=str(asset_dir), delete=False,
        ) as list_file:
            yaml.safe_dump(names, list_file, sort_keys=False)
            group_list_path = Path(list_file.name)

        try:
            group_list_arg = str(group_list_path.relative_to(Path(asset_root).resolve()))
            with tempfile.TemporaryDirectory(prefix="collect_object_partical_") as temp_dir:
                worker_output = Path(temp_dir) / "data"
                worker_output.mkdir()
                command = [
                    sys.executable, "-u", "-m", "perception.collect_object_partical_points",
                    "--output", str(worker_output),
                    "--num-envs", str(ARGS.num_envs),
                    "--samples-per-object", str(samples_per_object),
                    "--shard-size", str(ARGS.shard_size),
                    "--bbox-jitter-px", str(ARGS.bbox_jitter_px),
                    "--base-noise-xyz", *[str(value) for value in ARGS.base_noise_xyz],
                    "--max-attempts-per-sample", str(ARGS.max_attempts_per_sample),
                    "--seed", str(ARGS.seed + start),
                    "--width", str(ARGS.width),
                    "--height", str(ARGS.height),
                    "--fov-deg", str(ARGS.fov_deg),
                    "--_worker",
                    "--_asset-offset", str(start),
                    "--_object-list", group_list_arg,
                    *ARGS.hydra_argv,
                ]
                subprocess.run(command, check=True)
                worker_shards = sorted(worker_output.glob("shard_*.npz"))
                if not worker_shards:
                    raise RuntimeError("Object group worker produced no shard files")
                for source in worker_shards:
                    destination = output_dir / "shard_{:06d}.npz".format(next_shard)
                    source.replace(destination)
                    next_shard += 1
            existing_counts[start:end] = samples_per_object
            total_collected += len(names) * samples_per_object
            print("Finished group {}/{}; collected {}/{} samples.".format(
                group_number, group_count, total_collected, len(asset_names) * samples_per_object
            ), flush=True)
        finally:
            group_list_path.unlink(missing_ok=True)

    return total_collected


def _capture_env_group(
    cfg, asset_names, global_asset_ids, samples_per_object, output_dir,
    samples, shard_index, collected, rng, torch, gymtorch, isaacgymenvs,
    total_samples, group_number, group_count,
):
    cfg.num_envs = len(asset_names)
    cfg.task.env.numEnvs = len(asset_names)
    print(
        "Loading object group {}/{} ({} objects, {} parallel envs)...".format(
            group_number, group_count, len(asset_names), len(asset_names)
        ),
        flush=True,
    )
    env = isaacgymenvs.make(
        cfg.seed, cfg.task_name, len(asset_names), cfg.sim_device, cfg.rl_device,
        cfg.graphics_device_id, True, cfg.multi_gpu, False, cfg.force_render, cfg,
    )
    print("Simulator ready; capturing RGB-D frames...", flush=True)
    local_counts = np.zeros(len(asset_names), dtype=np.int32)
    attempts = np.zeros(len(asset_names), dtype=np.int32)
    env_ids = torch.arange(len(asset_names), dtype=torch.long, device=env.device)
    camera_handles = env.camera_handles
    intrinsics = np.asarray(env.camera_cfg["camera_1"]["intrinsics"], dtype=np.float32)
    base_xyz = None

    try:
        env.reset_idx(env_ids)
        env.gym.refresh_actor_root_state_tensor(env.sim)
        base_xyz = env.root_state_tensor[env.robot_indices, :3].clone()
        while np.any(local_counts < samples_per_object):
            env.reset_idx(env_ids)
            env.gym.refresh_actor_root_state_tensor(env.sim)
            base_delta = rng.uniform(
                -1, 1, size=(len(asset_names), 3)
            ).astype(np.float32) * np.asarray(ARGS.base_noise_xyz, dtype=np.float32)
            env.root_state_tensor[env.robot_indices, :3] = base_xyz + torch.as_tensor(
                base_delta, device=env.device, dtype=torch.float32
            )
            robot_indices = env.robot_indices.to(torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.root_state_tensor),
                gymtorch.unwrap_tensor(robot_indices),
                len(robot_indices),
            )
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            try:
                rgb_batch = [
                    tensor.cpu().numpy()[..., :3].copy()
                    for tensor in env.camera_rgb_tensor_lists[0]
                ]
                depth_batch = [tensor.cpu().numpy().copy() for tensor in env.camera_depth_tensor_lists[0]]
                seg_batch = [tensor.cpu().numpy().copy() for tensor in env.camera_seg_tensor_lists[0]]
            finally:
                env.gym.end_access_image_tensors(env.sim)

            env.gym.refresh_actor_root_state_tensor(env.sim)
            for env_index, global_asset_id in enumerate(global_asset_ids):
                if local_counts[env_index] >= samples_per_object:
                    continue
                attempts[env_index] += 1
                if attempts[env_index] > ARGS.max_attempts_per_sample:
                    raise RuntimeError(
                        "Could not collect sample {} for {} after {} captures".format(
                            local_counts[env_index], asset_names[env_index], attempts[env_index] - 1
                        )
                    )

                camera_transform = env.gym.get_camera_transform(
                    env.sim, env.envs[env_index], camera_handles[env_index]
                )
                camera_pose = np.array([
                    camera_transform.p.x, camera_transform.p.y, camera_transform.p.z,
                    camera_transform.r.x, camera_transform.r.y,
                    camera_transform.r.z, camera_transform.r.w,
                ], dtype=np.float32)
                object_pose = env.root_state_tensor[env.object_indices[env_index], :7].cpu().numpy().copy()
                points_local = np.asarray(env.object_pcls[env.object_asset_ids[env_index]], dtype=np.float32)
                points_world = points_local @ _quat_matrix(object_pose[3:]).T + object_pose[:3]
                points_camera = _world_to_cv(points_world, camera_pose)
                box = _box_from_points(
                    points_camera, intrinsics, ARGS.width, ARGS.height,
                    ARGS.bbox_jitter_px, rng,
                )
                object_mask = (seg_batch[env_index] == 2).astype(np.uint8)
                if box is None or not np.any(object_mask):
                    continue

                point_count = 512
                if len(points_camera) >= point_count:
                    selected = rng.choice(len(points_camera), point_count, replace=False)
                else:
                    selected = rng.choice(len(points_camera), point_count, replace=True)
                samples.append({
                    "rgb": rgb_batch[env_index],
                    "depth": depth_batch[env_index],
                    "bbox": box,
                    "object_mask": object_mask,
                    "object_asset_id": np.int32(global_asset_id),
                    "object_sample_index": np.int32(local_counts[env_index]),
                    "points_camera": points_camera[selected].astype(np.float32),
                    "pose_camera": _object_pose_cv(object_pose, camera_pose).astype(np.float32),
                })
                local_counts[env_index] += 1
                attempts[env_index] = 0
                collected += 1
                if len(samples) >= ARGS.shard_size:
                    _save_shard(output_dir, shard_index, samples)
                    shard_index += 1
                    samples.clear()

            print("Collected {}/{} samples".format(collected, total_samples))
    finally:
        env.gym.destroy_sim(env.sim)
        if samples:
            _save_shard(output_dir, shard_index, samples)
            samples.clear()
            shard_index += 1

    return shard_index, collected, intrinsics


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    print("Preparing balanced RGB-D collection...", flush=True)
    output_dir = Path(ARGS.output)
    if (output_dir / "metadata.json").exists():
        raise FileExistsError("{} already contains a completed dataset".format(output_dir))
    if ARGS._worker and list(output_dir.glob("shard_*.npz")):
        raise FileExistsError("Worker output directory already contains shards: {}".format(output_dir))

    if cfg.task.hand_config.name != "g1_upper_right_wuji":
        raise ValueError("Collection is configured for hand=g1_upper_right_wuji")
    if ARGS.num_envs < 1 or ARGS.shard_size < 1 or ARGS.max_attempts_per_sample < 1:
        raise ValueError("--num-envs, --shard-size, and --max-attempts-per-sample must be positive")
    if ARGS.width < 1 or ARGS.height < 1 or not 0 < ARGS.fov_deg < 180:
        raise ValueError("--width and --height must be positive, and --fov-deg must be between 0 and 180")

    import yaml

    asset_root = Path(cfg.task.env.asset.assetRoot).resolve()
    original_list = (
        str(ARGS._object_list)
        if ARGS._worker and ARGS._object_list is not None
        else str(cfg.task.env.asset.multiObjectList)
    )
    if ARGS._worker:
        cfg.task.env.asset.multiObjectList = original_list
    list_path = asset_root / original_list
    if not list_path.is_file():
        raise FileNotFoundError("Object list not found: {}".format(list_path))
    asset_names = sorted(yaml.safe_load(list_path.read_text()))
    if not asset_names:
        raise ValueError("The configured object list is empty")
    if ARGS.samples is not None and ARGS.samples < 1:
        raise ValueError("--samples must be positive")

    if ARGS.samples_per_object is not None:
        if ARGS.samples_per_object < 1:
            raise ValueError("--samples-per-object must be positive")
        samples_per_object = ARGS.samples_per_object
        if ARGS.samples is not None and ARGS.samples != samples_per_object * len(asset_names):
            raise ValueError(
                "--samples must equal object_count * --samples-per-object ({} * {} = {}), got {}".format(
                    len(asset_names), samples_per_object,
                    len(asset_names) * samples_per_object, ARGS.samples,
                )
            )
    elif ARGS.samples is None:
        samples_per_object = 100
    else:
        if ARGS.samples < 1 or ARGS.samples % len(asset_names):
            raise ValueError(
                "--samples must be positive and divisible by the {} configured objects".format(len(asset_names))
            )
        samples_per_object = ARGS.samples // len(asset_names)

    total_samples = len(asset_names) * samples_per_object
    group_size = min(ARGS.num_envs, len(asset_names))
    group_count = (len(asset_names) + group_size - 1) // group_size
    if ARGS._worker:
        print(
            "Dataset plan: {} objects, {} samples/object ({} total), {} envs in this worker.".format(
                len(asset_names), samples_per_object, total_samples, len(asset_names)
            ),
            flush=True,
        )
    cfg.seed = ARGS.seed
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
    cfg.task.env.asset.randomizeObjectAssets = False
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

    if not ARGS._worker:
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_group_workers(
            cfg, asset_root, asset_names, original_list, output_dir,
            samples_per_object, yaml,
        )
        metadata = {
            "width": ARGS.width,
            "height": ARGS.height,
            "intrinsics": np.asarray(camera.intrinsics, dtype=np.float32).tolist(),
            "depth_convention": "raw Isaac Gym IMAGE_DEPTH; negative meters",
            "point_frame": "camera CV: right, down, forward; meters",
            "pose_frame": "camera CV; translation meters, quaternion xyzw",
            "point_cloud_label_type": "full_asset_surface",
            "object_urdf_files": asset_names,
            "point_cloud_label": (
                "512 points from the complete target asset point cloud transformed to camera CV coordinates, "
                "including surfaces occluded in the RGB-D frame"
            ),
            "bbox_order": "xmin, ymin, xmax, ymax pixels",
            "camera_horizontal_fov_deg": ARGS.fov_deg,
            "camera_depth_range": [0.1, 5.0],
            "camera_position_head_link": list(camera.position),
            "camera_rpy_deg_head_link": [0.0, 45.0, 0.0],
            "camera_quaternion_head_link_xyzw": list(camera.quaternion),
            "samples": total_samples,
            "samples_per_object": samples_per_object,
            "uniform_object_sampling": True,
            "object_asset_id_order": "index in object_urdf_files",
            "object_sample_index": "zero-based index within each object's samples",
            "collection_workers": "one isolated Isaac Gym process per object group",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print("Finished dataset: {} samples saved to {}".format(total_samples, output_dir), flush=True)
        return

    # Isaac Gym must be imported before torch in each simulation worker.
    from isaacgym import gymtorch
    import torch
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import tasks  # noqa: F401

    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(ARGS.seed)
    samples, shard_index, collected = [], 0, 0
    intrinsics = None

    asset_dir = list_path.parent
    try:
        for group_number, start in enumerate(range(0, len(asset_names), group_size), start=1):
            end = min(start + group_size, len(asset_names))
            names = asset_names[start:end]
            group_list_path = None
            if len(names) < len(asset_names):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", prefix=".collect_object_partical_",
                    dir=str(asset_dir), delete=False,
                ) as list_file:
                    yaml.safe_dump(names, list_file, sort_keys=False)
                    group_list_path = Path(list_file.name)
                cfg.task.env.asset.multiObjectList = str(
                    group_list_path.relative_to(Path(asset_root).resolve())
                )
            else:
                cfg.task.env.asset.multiObjectList = original_list

            try:
                shard_index, collected, intrinsics = _capture_env_group(
                    cfg, names,
                    list(range(ARGS._asset_offset + start, ARGS._asset_offset + end)),
                    samples_per_object,
                    output_dir, samples, shard_index, collected, rng, torch,
                    gymtorch, isaacgymenvs, total_samples, group_number, group_count,
                )
            finally:
                if group_list_path is not None:
                    group_list_path.unlink(missing_ok=True)

        _save_shard(output_dir, shard_index, samples)
        samples.clear()
        metadata = {
            "width": ARGS.width,
            "height": ARGS.height,
            "intrinsics": intrinsics.tolist(),
            "depth_convention": "raw Isaac Gym IMAGE_DEPTH; negative meters",
            "point_frame": "camera CV: right, down, forward; meters",
            "pose_frame": "camera CV; translation meters, quaternion xyzw",
            "point_cloud_label_type": "full_asset_surface",
            "object_urdf_files": asset_names,
            "point_cloud_label": (
                "512 points from the complete target asset point cloud transformed to camera CV coordinates, "
                "including surfaces occluded in the RGB-D frame"
            ),
            "bbox_order": "xmin, ymin, xmax, ymax pixels",
            "camera_horizontal_fov_deg": ARGS.fov_deg,
            "camera_depth_range": [0.1, 5.0],
            "camera_position_head_link": list(camera.position),
            "camera_rpy_deg_head_link": [0.0, 45.0, 0.0],
            "camera_quaternion_head_link_xyzw": list(camera.quaternion),
            "samples": total_samples,
            "samples_per_object": samples_per_object,
            "uniform_object_sampling": True,
            "object_asset_id_order": "index in object_urdf_files",
            "object_sample_index": "zero-based index within each object's samples",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    finally:
        if samples:
            _save_shard(output_dir, shard_index, samples)


if __name__ == "__main__":
    main()
