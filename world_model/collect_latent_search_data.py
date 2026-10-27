"""Record camera, point-cloud, and robot-state sequences during latent grasp search."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ae-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--output", default="datasets/latent_search_rgbd")
    parser.add_argument("--object-index", type=int, default=None,
                        help="object row in the grasp dataset; default selects one at random")
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--update-rate", type=float, default=0.7)
    parser.add_argument("--max-base-abs", type=float, default=4.0)
    parser.add_argument(
        "--max-object-point-abs",
        type=float,
        default=2.0,
        help="maximum absolute robot-base coordinate allowed for an object point (meters)",
    )
    parser.add_argument(
        "--max-hand-point-abs",
        type=float,
        default=2.0,
        help="maximum absolute robot-base coordinate allowed for a hand point (meters)",
    )
    parser.add_argument(
        "--max-point-step",
        type=float,
        default=1.0,
        help="maximum same-point displacement between consecutive frames (meters)",
    )
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--post-lift-height", type=float, default=0.10)
    parser.add_argument("--post-lift-steps", type=int, default=10)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--lift-weight", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=5.0)
    parser.add_argument("--prior-penalty", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument("--camera-position", type=float, nargs=3, default=(0.1, 0.0, 0.45))
    parser.add_argument("--camera-rpy-deg", type=float, nargs=3, default=(0.0, 45.0, 0.0))
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed; omitted selects a fresh seed for this run")
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args, hydra_argv


ARGS, HYDRA_ARGV = _parse_args()


def _audit_point_cloud(points, name, max_abs, max_step):
    points = np.asarray(points, dtype=np.float32)
    reasons = []
    if points.ndim != 3 or points.shape[-1] != 3:
        return ["{}_shape".format(name)]
    if not np.isfinite(points).all():
        reasons.append("{}_nonfinite".format(name))
        return reasons
    max_coordinate = float(np.abs(points).max()) if points.size else 0.0
    if max_coordinate > max_abs:
        reasons.append("{}_abs>{:.6g}".format(name, max_abs))
    if points.shape[0] > 1:
        frame_step = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        max_frame_step = float(frame_step.max()) if frame_step.size else 0.0
        if max_frame_step > max_step:
            reasons.append("{}_step>{:.6g}".format(name, max_step))
    return reasons


def _audit_pose_array(pose, name):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim != 2 or pose.shape[-1] != 7:
        return ["{}_shape".format(name)]
    if not np.isfinite(pose).all():
        return ["{}_nonfinite".format(name)]
    quaternion_norm = np.linalg.norm(pose[:, 3:], axis=-1)
    if np.any(quaternion_norm < 1e-6):
        return ["{}_invalid_quaternion".format(name)]
    return []


def _deproject_masked_depth(depth, mask, intrinsics, num_points, rng, near, far):
    """Return a fixed-size partial cloud in camera-CV coordinates (right, down, forward)."""
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    valid = mask & np.isfinite(depth) & (depth < -1e-6)
    distance = -depth
    valid &= (distance >= near) & (distance <= far)
    indices = np.flatnonzero(valid)
    cloud = np.zeros((num_points, 3), dtype=np.float32)
    if indices.size == 0:
        return cloud, 0

    selected = rng.choice(indices, size=min(indices.size, num_points), replace=False)
    height, width = depth.shape
    v, u = np.divmod(selected, width)
    z = distance.reshape(-1)[selected]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    cloud[:len(selected)] = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)
    if len(selected) < num_points:
        fill = rng.choice(len(selected), size=num_points - len(selected), replace=True)
        cloud[len(selected):] = cloud[fill]
    return cloud, int(indices.size)


class _IterationRecorder:
    def __init__(self, output_dir, iteration, env, frame_count, point_count, intrinsics,
                 near, far, seed, hand_mesh_points, robot_base_body_index, robot_base_link,
                 max_object_point_abs, max_hand_point_abs, max_point_step):
        self.output_dir = Path(output_dir) / "iteration_{:03d}".format(iteration)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env = env
        self.frame_count = frame_count
        self.point_count = point_count
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.hand_mesh_points = hand_mesh_points
        self.robot_base_body_index = int(robot_base_body_index)
        self.robot_base_link = str(robot_base_link)
        self.max_object_point_abs = float(max_object_point_abs)
        self.max_hand_point_abs = float(max_hand_point_abs)
        self.max_point_step = float(max_point_step)
        self.near = near
        self.far = far
        self.rng = np.random.default_rng(seed)
        self.next_step = 0
        self.phases = np.zeros(frame_count, dtype=np.uint8)
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix=".capture_", dir=str(Path(output_dir))
        )
        temp_path = Path(self.temp_dir.name)
        population = env.num_envs
        height = int(env.camera_cfg["camera_1"]["height"])
        width = int(env.camera_cfg["camera_1"]["width"])
        dofs = env.robot_dof_pos.shape[-1]
        actions = env.num_actions
        shapes = {
            "rgb": ((population, frame_count, height, width, 3), np.uint8),
            "depth": ((population, frame_count, height, width), np.float32),
            "object_mask": ((population, frame_count, height, width), np.uint8),
            "hand_mask": ((population, frame_count, height, width), np.uint8),
            "object_points_camera": ((population, frame_count, point_count, 3), np.float32),
            "hand_points_base": ((population, frame_count, point_count, 3), np.float32),
            "point_counts": ((population, frame_count, 3), np.int32),
            "camera_pose_world": ((population, frame_count, 7), np.float32),
            "object_pose_world": ((population, frame_count, 7), np.float32),
            "robot_base_pose_world": ((population, frame_count, 7), np.float32),
            "robot_qpos": ((population, frame_count, dofs), np.float32),
            "robot_qvel": ((population, frame_count, dofs), np.float32),
            "actions": ((population, frame_count, actions), np.float32),
        }
        self.arrays = {
            key: np.lib.format.open_memmap(
                temp_path / (key + ".npy"), mode="w+", dtype=dtype, shape=shape
            )
            for key, (shape, dtype) in shapes.items()
        }

    def capture(self, step, action, phase):
        if step != self.next_step or step >= self.frame_count:
            raise RuntimeError("Unexpected capture frame {} (expected {})".format(step, self.next_step))
        env = self.env
        gym = env.gym
        gym.start_access_image_tensors(env.sim)
        try:
            rgb = np.stack([tensor.cpu().numpy()[..., :3] for tensor in env.camera_rgb_tensor_lists[0]])
            depth = np.stack([tensor.cpu().numpy() for tensor in env.camera_depth_tensor_lists[0]])
            segmentation = np.stack([tensor.cpu().numpy() for tensor in env.camera_seg_tensor_lists[0]])
        finally:
            gym.end_access_image_tensors(env.sim)

        object_mask = segmentation == 2
        hand_mask = segmentation == 3
        self.arrays["rgb"][:, step] = rgb
        self.arrays["depth"][:, step] = depth
        self.arrays["object_mask"][:, step] = object_mask.astype(np.uint8)
        self.arrays["hand_mask"][:, step] = hand_mask.astype(np.uint8)
        self.arrays["actions"][:, step] = action.detach().cpu().numpy()
        self.arrays["robot_qpos"][:, step] = env.robot_dof_pos[:, env.robot_dof_indices].detach().cpu().numpy()
        self.arrays["robot_qvel"][:, step] = env.robot_dof_vel[:, env.robot_dof_indices].detach().cpu().numpy()
        self.arrays["object_pose_world"][:, step] = (
            env.root_state_tensor[env.object_indices, :7].detach().cpu().numpy()
        )
        hand_body_states = env.rigid_body_states[:, [entry[1] for entry in self.hand_mesh_points], :7]
        hand_body_states = hand_body_states.detach().cpu().numpy()
        robot_base_states = env.rigid_body_states[:, self.robot_base_body_index, :7]
        robot_base_states = robot_base_states.detach().cpu().numpy()

        for env_id in range(env.num_envs):
            camera_transform = gym.get_camera_transform(
                env.sim, env.envs[env_id], env.camera_handles[env_id]
            )
            camera_pose = np.asarray([
                camera_transform.p.x, camera_transform.p.y, camera_transform.p.z,
                camera_transform.r.x, camera_transform.r.y, camera_transform.r.z,
                camera_transform.r.w,
            ], dtype=np.float32)
            self.arrays["camera_pose_world"][env_id, step] = camera_pose
            object_cloud, object_count = _deproject_masked_depth(
                depth[env_id], object_mask[env_id], self.intrinsics,
                self.point_count, self.rng, self.near, self.far,
            )
            _, hand_count = _deproject_masked_depth(
                depth[env_id], hand_mask[env_id], self.intrinsics,
                self.point_count, self.rng, self.near, self.far,
            )
            robot_base_pose = np.asarray(robot_base_states[env_id], dtype=np.float32)
            self.arrays["robot_base_pose_world"][env_id, step] = robot_base_pose
            base_rotation = _rotation_from_xyzw(robot_base_pose[3:])
            hand_cloud_base = []
            for body_slot, (_, _, points_link) in enumerate(self.hand_mesh_points):
                body_state = hand_body_states[env_id, body_slot]
                body_rotation = _rotation_from_xyzw(body_state[3:7])
                points_world = points_link @ body_rotation.T + body_state[:3]
                hand_cloud_base.append((points_world - robot_base_pose[:3]) @ base_rotation)
            hand_cloud_base = np.concatenate(hand_cloud_base, axis=0).astype(np.float32)
            if len(hand_cloud_base) != self.point_count:
                raise RuntimeError("Hand mesh sampler returned an unexpected point count")
            self.arrays["object_points_camera"][env_id, step] = object_cloud
            self.arrays["hand_points_base"][env_id, step] = hand_cloud_base
            self.arrays["point_counts"][env_id, step] = (
                object_count, hand_count, len(hand_cloud_base)
            )

        self.phases[step] = 0 if phase == "trajectory" else 1
        self.next_step += 1

    def save(self, base, metrics, iteration, robot_dof_names, object_points_local,
             decoded, object_frame_rpy):
        from scipy.spatial.transform import Rotation

        if self.next_step != self.frame_count:
            raise RuntimeError("Recorded {} frames, expected {}".format(self.next_step, self.frame_count))
        for array in self.arrays.values():
            array.flush()
        metric_names = (
            "fitness", "max_lift", "final_lift", "lift_score", "success_seen",
            "lifted", "collision", "arm_joint_motion", "arm_target_motion",
            "arm_target_error", "eef_target_lift", "eef_actual_lift", "valid_success",
        )
        floating_fields = (
            "object_points_camera", "hand_points_base", "camera_pose_world",
            "object_pose_world", "robot_base_pose_world", "robot_qpos", "robot_qvel",
            "actions",
        )
        valid_candidates = np.zeros(self.env.num_envs, dtype=bool)
        rejected = []
        for candidate in range(self.env.num_envs):
            path = self.output_dir / "candidate_{:04d}.npz".format(candidate)
            payload = {
                key: np.asarray(value[candidate]).copy()
                for key, value in self.arrays.items()
            }
            payload.update({
                "frame_phase": self.phases.copy(),
                "base_u": base[candidate].detach().cpu().numpy(),
                "latent_z": metrics["latent"][candidate].detach().cpu().numpy(),
                "decoded_trajectory": decoded[candidate].detach().cpu().numpy(),
                "object_frame_rpy": object_frame_rpy[candidate].detach().cpu().numpy(),
                "world_trajectory": metrics["world_targets"][candidate].detach().cpu().numpy(),
                "hand_trajectory": metrics["hand_targets"][candidate].detach().cpu().numpy(),
                "robot_base_link": np.asarray(self.robot_base_link, dtype=np.str_),
                "robot_dof_names": np.asarray(robot_dof_names, dtype=np.str_),
                "iteration": np.asarray(iteration, dtype=np.int32),
                "candidate_index": np.asarray(candidate, dtype=np.int32),
            })
            for name in metric_names:
                payload[name] = metrics[name][candidate].detach().cpu().numpy()

            reasons = []
            reasons.extend(_audit_pose_array(payload["object_pose_world"], "object_pose_world"))
            reasons.extend(_audit_pose_array(payload["robot_base_pose_world"], "robot_base_pose_world"))
            for name in floating_fields + metric_names + (
                "base_u", "latent_z", "decoded_trajectory", "object_frame_rpy",
                "world_trajectory", "hand_trajectory",
            ):
                value = np.asarray(payload[name])
                if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                    reasons.append("{}_nonfinite".format(name))
            if reasons:
                rejected.append({"candidate_index": candidate, "reasons": sorted(set(reasons))})
                continue

            object_poses = payload["object_pose_world"]
            base_poses = payload["robot_base_pose_world"]
            object_rotations = Rotation.from_quat(object_poses[:, 3:]).as_matrix().astype(np.float32)
            base_rotations = Rotation.from_quat(base_poses[:, 3:]).as_matrix().astype(np.float32)
            object_points_world = (
                np.asarray(object_points_local, dtype=np.float32)[None, :, :]
                @ np.swapaxes(object_rotations, 1, 2)
                + object_poses[:, None, :3]
            )
            payload["object_points_base"] = (
                object_points_world - base_poses[:, None, :3]
            ) @ base_rotations
            reasons.extend(_audit_point_cloud(
                payload["object_points_base"], "object_points_base",
                self.max_object_point_abs, self.max_point_step,
            ))
            reasons.extend(_audit_point_cloud(
                payload["hand_points_base"], "hand_points_base",
                self.max_hand_point_abs, self.max_point_step,
            ))
            if reasons:
                rejected.append({"candidate_index": candidate, "reasons": sorted(set(reasons))})
                continue

            np.savez_compressed(path, **payload)
            valid_candidates[candidate] = True

        self.arrays.clear()
        self.temp_dir.cleanup()
        accepted_count = int(valid_candidates.sum())
        print("Trajectory audit iteration {:03d}: accepted {}/{}, rejected {}".format(
            iteration, accepted_count, self.env.num_envs, len(rejected)
        ))
        for item in rejected:
            print("  rejected candidate_{:04d}: {}".format(
                item["candidate_index"], ", ".join(item["reasons"])
            ))
        (self.output_dir / "trajectory_audit.json").write_text(json.dumps({
            "iteration": int(iteration),
            "accepted_count": accepted_count,
            "candidate_count": int(self.env.num_envs),
            "thresholds_m": {
                "max_object_point_abs": self.max_object_point_abs,
                "max_hand_point_abs": self.max_hand_point_abs,
                "max_same_point_step": self.max_point_step,
            },
            "rejected": rejected,
        }, indent=2))
        return {
            "valid_candidates": valid_candidates,
            "accepted": accepted_count,
            "rejected": rejected,
        }


def _set_hand_segmentation(env, hand_segmentation_id=3):
    gym = env.gym
    hand_config = env.hand_specific_cfg
    hand_prefixes = {
        name.rsplit("_joint", 1)[0] + "_"
        for name in hand_config["active_hand_dof_names"]
    }
    palm_link = hand_config["palm_link"]
    hand_body_names = None
    for env_ptr in env.envs:
        robot_actor = gym.find_actor_handle(env_ptr, "robot")
        body_names = gym.get_actor_rigid_body_names(env_ptr, robot_actor)
        hand_indices = [
            index for index, name in enumerate(body_names)
            if name == palm_link or any(name.startswith(prefix) for prefix in hand_prefixes)
        ]
        if not hand_indices:
            raise RuntimeError("Could not find hand rigid bodies for segmentation")
        for body_index in hand_indices:
            gym.set_rigid_body_segmentation_id(
                env_ptr, robot_actor, body_index, hand_segmentation_id
            )
        hand_body_names = [body_names[index] for index in hand_indices]
    return hand_body_names


def _resolve_robot_base_body_index(env, gymapi, link_name):
    actor = env.gym.find_actor_handle(env.envs[0], "robot")
    body_handle = env.gym.find_actor_rigid_body_handle(env.envs[0], actor, str(link_name))
    if body_handle < 0:
        raise RuntimeError("Robot base link {!r} was not found in the robot asset".format(link_name))
    body_index = env.gym.get_actor_rigid_body_index(
        env.envs[0], actor, body_handle, gymapi.DOMAIN_ENV
    )
    if body_index < 0 or body_index >= env.rigid_body_states.shape[1]:
        raise RuntimeError("Robot base link {!r} has an invalid rigid-body index".format(link_name))
    return int(body_index)


def _rotation_from_xyzw(quaternion):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(quaternion, dtype=np.float64)).as_matrix()


def _sample_hand_mesh_points(env, gymapi, urdf_path, hand_body_names, point_count, seed):
    import trimesh
    from scipy.spatial.transform import Rotation

    urdf_path = Path(urdf_path).resolve()
    root = ET.parse(str(urdf_path)).getroot()
    wanted_links = set(hand_body_names)
    mesh_records = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if link_name not in wanted_links:
            continue
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            if mesh_element is None:
                continue
            mesh_path = Path(mesh_element.get("filename"))
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            scale = np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
            vertices = np.asarray(mesh.vertices, dtype=np.float64) * scale
            triangles = vertices[np.asarray(mesh.faces, dtype=np.int64)]
            face_areas = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=-1,
            )
            origin = visual.find("origin")
            xyz = np.zeros(3) if origin is None else np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.zeros(3) if origin is None else np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            visual_rotation = Rotation.from_euler("xyz", rpy).as_matrix()
            mesh_records.append({
                "link_name": link_name,
                "triangles": triangles,
                "face_areas": face_areas,
                "area": float(face_areas.sum()),
                "visual_rotation": visual_rotation,
                "visual_translation": xyz,
            })

    present_links = {record["link_name"] for record in mesh_records}
    missing_links = sorted(wanted_links - present_links)
    if missing_links:
        raise RuntimeError("No visual mesh found for hand links: {}".format(missing_links))
    total_area = sum(record["area"] for record in mesh_records)
    if total_area <= 0:
        raise RuntimeError("Wuji hand visual meshes have no surface area")

    raw_counts = np.asarray([
        point_count * record["area"] / total_area for record in mesh_records
    ])
    sample_counts = np.floor(raw_counts).astype(np.int64)
    remainder = point_count - int(sample_counts.sum())
    if remainder:
        order = np.argsort(raw_counts - sample_counts)[::-1]
        sample_counts[order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    points_by_link = {name: [] for name in hand_body_names}
    for record, sample_count in zip(mesh_records, sample_counts):
        if sample_count == 0:
            continue
        probabilities = record["face_areas"] / record["area"]
        face_indices = rng.choice(len(probabilities), size=sample_count, p=probabilities)
        triangles = record["triangles"][face_indices]
        root_u = np.sqrt(rng.random(sample_count))
        bary_v = rng.random(sample_count)
        points_visual = (
            (1.0 - root_u)[:, None] * triangles[:, 0]
            + (root_u * (1.0 - bary_v))[:, None] * triangles[:, 1]
            + (root_u * bary_v)[:, None] * triangles[:, 2]
        )
        points_link = (
            points_visual @ record["visual_rotation"].T + record["visual_translation"]
        ).astype(np.float32)
        points_by_link[record["link_name"]].append(points_link)

    actor = env.gym.find_actor_handle(env.envs[0], "robot")
    result = []
    for link_name in hand_body_names:
        body_handle = env.gym.find_actor_rigid_body_handle(
            env.envs[0], actor, link_name
        )
        body_index = env.gym.get_actor_rigid_body_index(
            env.envs[0], actor, body_handle, gymapi.DOMAIN_ENV
        )
        link_points = points_by_link[link_name]
        points_link = np.concatenate(link_points, axis=0) if link_points else np.empty((0, 3), np.float32)
        result.append((link_name, int(body_index), points_link))
    if sum(len(entry[2]) for entry in result) != point_count:
        raise RuntimeError("Could not allocate the requested number of hand mesh points")
    return result


def _import_search_module(selected_index):
    search_argv = [
        "--dataset", ARGS.dataset,
        "--ae-checkpoint", ARGS.ae_checkpoint,
        "--flow-checkpoint", ARGS.flow_checkpoint,
        "--object-index", str(selected_index),
        "--population-size", str(ARGS.population_size),
        "--iterations", str(ARGS.iterations),
        "--elite-fraction", str(ARGS.elite_fraction),
        "--initial-std", str(ARGS.initial_std),
        "--min-std", str(ARGS.min_std),
        "--max-std", str(ARGS.max_std),
        "--update-rate", str(ARGS.update_rate),
        "--max-base-abs", str(ARGS.max_base_abs),
        "--hold-steps", str(ARGS.hold_steps),
        "--post-lift-height", str(ARGS.post_lift_height),
        "--post-lift-steps", str(ARGS.post_lift_steps),
        "--lift-height", str(ARGS.lift_height),
        "--lift-weight", str(ARGS.lift_weight),
        "--success-bonus", str(ARGS.success_bonus),
        "--collision-penalty", str(ARGS.collision_penalty),
        "--prior-penalty", str(ARGS.prior_penalty),
        *HYDRA_ARGV,
    ]
    original_argv = sys.argv
    sys.argv = [original_argv[0], *search_argv]
    try:
        return importlib.import_module("testing.search_latent_grasp")
    finally:
        sys.argv = original_argv


def _configure_camera(cfg, gymapi):
    cfg.headless = True
    cfg.force_render = False
    cfg.task.env.enableCameraSensors = True
    cfg.task.env.enablePointCloud = True
    cfg.task.env.render.enable = True
    cfg.task.env.render.camera_ids = [1]
    cfg.task.env.render.data_type = "rgb,depth,seg"
    cfg.task.env.render.randomize = False
    camera = cfg.task.env.camera_config.camera_1
    camera.mount = "head_link"
    camera.width = ARGS.width
    camera.height = ARGS.height
    camera.depth_range = [ARGS.near, ARGS.far]
    focal = ARGS.width / (2.0 * math.tan(math.radians(ARGS.fov_deg) / 2.0))
    camera.intrinsics = [
        [focal, 0.0, ARGS.width / 2.0],
        [0.0, focal, ARGS.height / 2.0],
        [0.0, 0.0, 1.0],
    ]
    roll, pitch, yaw = (math.radians(value) for value in ARGS.camera_rpy_deg)
    quaternion = gymapi.Quat.from_euler_zyx(yaw, pitch, roll)
    OmegaConf.update(cfg, "task.env.camera_config.camera_1.position",
                     list(ARGS.camera_position), force_add=True)
    OmegaConf.update(cfg, "task.env.camera_config.camera_1.quaternion",
                     [quaternion.x, quaternion.y, quaternion.z, quaternion.w], force_add=True)


def _select_object_index():
    with np.load(ARGS.dataset, allow_pickle=False) as data:
        count = len(data["object_label"])
    if count < 1:
        raise ValueError("The grasp dataset contains no objects")
    if ARGS.object_index is None:
        return int(np.random.default_rng(ARGS.seed).integers(count))
    if not 0 <= ARGS.object_index < count:
        raise IndexError("--object-index must be in [0, {})".format(count))
    return ARGS.object_index


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig):
    from isaacgym import gymapi
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_seed
    from isaacgymenvs.tasks.base import vec_task
    import tasks  # noqa: F401

    if ARGS.population_size < 1 or ARGS.iterations < 1 or ARGS.num_points < 1:
        raise ValueError("population size, iterations, and point count must be positive")
    if min(ARGS.max_object_point_abs, ARGS.max_hand_point_abs, ARGS.max_point_step) <= 0:
        raise ValueError("trajectory audit limits must be positive")
    if ARGS.width < 32 or ARGS.height < 32 or ARGS.near <= 0 or ARGS.far <= ARGS.near:
        raise ValueError("invalid camera dimensions or depth range")
    if cfg.task.env.armController != "pose":
        raise ValueError("Latent trajectory search requires task.env.armController=pose")
    if cfg.task.hand_config.name != "g1_upper_right_wuji":
        raise ValueError("This collector currently segments the g1_upper_right_wuji hand")
    if cfg.graphics_device_id < 0:
        raise ValueError("RGB-D collection requires a valid graphics device in headless mode")

    global torch
    import torch
    if ARGS.seed is None:
        ARGS.seed = int(np.random.SeedSequence().generate_state(1)[0])
    search = _import_search_module(_select_object_index())
    search.torch = torch
    search.ARGS.seed = ARGS.seed
    label, object_pose_np = search._load_object_pose()

    output_dir = Path(ARGS.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("{} must be empty to start a new collection round".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    ae_checkpoint = torch.load(ARGS.ae_checkpoint, map_location="cpu", weights_only=False)
    flow_checkpoint = torch.load(ARGS.flow_checkpoint, map_location="cpu", weights_only=False)
    sequence_length = int(ae_checkpoint["config"]["sequence_length"])
    latent_dim = int(flow_checkpoint["config"]["latent_dim"])
    latent_shape = tuple(flow_checkpoint["config"]["latent_shape"])
    from memory.train_latent_flow import _make_flow
    from memory.train_trajectory_ae import _build_model
    from memory.train_trajectory_vqvae import rotation6d_to_matrix

    cfg.seed = ARGS.seed
    cfg.num_envs = ARGS.population_size
    cfg.task.env.numEnvs = ARGS.population_size
    cfg.task.env.episodeLength = (
        sequence_length + ARGS.post_lift_steps + ARGS.hold_steps + 1
    )
    cfg.task.env.activeArm = 1
    cfg.task.env.randomizeTrackingReference = False
    cfg.task.env.randomizeGraspPose = False
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.task.env.asset.multiObject = False
    OmegaConf.update(cfg, "task.env.asset.objectAssetFile", label, force_add=True)
    cfg.task.env.asset.useDistractorObjects = False
    _configure_camera(cfg, gymapi)
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

    env = None
    try:
        env = isaacgymenvs.make(
            cfg.seed, cfg.task_name, ARGS.population_size, cfg.sim_device,
            cfg.rl_device, cfg.graphics_device_id, True, cfg.multi_gpu, False,
            cfg.force_render, cfg,
        )
        device = torch.device(env.device)
        flow = _make_flow(flow_checkpoint["config"], device)
        flow.load_state_dict(flow_checkpoint["model"])
        flow.eval()
        ae = _build_model(ae_checkpoint["config"]).to(device)
        ae.load_state_dict(ae_checkpoint["model"])
        ae.eval()
        object_pose = torch.as_tensor(object_pose_np, dtype=torch.float32, device=device)
        dual_arm = bool(cfg.task.hand_config.get(
            "dualArm", cfg.task.hand_config.get("dualArmInspire", False)
        ))
        if dual_arm:
            env.active_arm_ids.fill_(False)
            offsets = cfg.task.hand_config.trajectoryEefOffsetByArm
            eef_offset = torch.as_tensor(offsets.right, dtype=torch.float32, device=device)
        else:
            offset = cfg.task.hand_config.get("trajectoryEefOffset", None)
            eef_offset = None if offset is None else torch.as_tensor(offset, dtype=torch.float32, device=device)

        hand_body_names = _set_hand_segmentation(env)
        robot_base_link = str(cfg.task.hand_config.get("robotBaseLink", ""))
        if not robot_base_link:
            raise ValueError("The selected hand configuration must define robotBaseLink")
        robot_base_body_index = _resolve_robot_base_body_index(env, gymapi, robot_base_link)
        hand_urdf_path = Path(env.asset_root) / str(cfg.task.hand_config.robotAssetFile)
        hand_mesh_points = _sample_hand_mesh_points(
            env, gymapi, hand_urdf_path, hand_body_names, ARGS.num_points,
            ARGS.seed + 1000,
        )
        intrinsics = np.asarray(env.camera_cfg["camera_1"]["intrinsics"], dtype=np.float32)
        sequence_steps = sequence_length + ARGS.post_lift_steps
        frame_count = sequence_steps + ARGS.hold_steps
        point_cloud_full_local = np.asarray(env.object_pcls[0], dtype=np.float32).copy()
        rng = np.random.default_rng(ARGS.seed)
        search_dim = latent_dim
        mean = torch.zeros(search_dim, dtype=torch.float32, device=device)
        std = torch.full_like(mean, float(ARGS.initial_std))
        elite_count = max(1, int(np.ceil(ARGS.population_size * ARGS.elite_fraction)))
        global_best = None
        iteration_audits = []

        print("Selected dataset object {}: {}".format(search.ARGS.object_index, label))
        print("Object pose is fixed across all candidates and CEM iterations.")
        print("Recording {} envs x {} iterations x {} action frames.".format(
            ARGS.population_size, ARGS.iterations, frame_count
        ))

        for iteration in range(1, ARGS.iterations + 1):
            base = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                ARGS.population_size, search_dim, device=device
            )
            base = base.clamp(-ARGS.max_base_abs, ARGS.max_base_abs)
            base[0] = mean.clamp(-ARGS.max_base_abs, ARGS.max_base_abs)
            recorder = _IterationRecorder(
                output_dir, iteration, env, frame_count, ARGS.num_points,
                intrinsics, ARGS.near, ARGS.far, ARGS.seed + iteration,
                hand_mesh_points, robot_base_body_index, robot_base_link,
                ARGS.max_object_point_abs, ARGS.max_hand_point_abs, ARGS.max_point_step,
            )
            metrics = search._evaluate_population(
                env, flow, ae, base, latent_shape, object_pose,
                rotation6d_to_matrix, False, eef_offset, latent_dim,
                frame_callback=recorder.capture,
            )
            audit = recorder.save(
                base, metrics, iteration, env.robot_dof_names,
                point_cloud_full_local, metrics["decoded"], metrics["object_frame_rpy"],
            )
            iteration_audits.append({
                "iteration": iteration,
                "accepted": audit["accepted"],
                "rejected": audit["rejected"],
            })
            valid_candidates = torch.as_tensor(
                audit["valid_candidates"], dtype=torch.bool, device=device
            )
            valid_count = int(valid_candidates.sum().item())
            if valid_count == 0:
                print("iter {:03d}/{:03d}: no valid trajectories; CEM distribution unchanged".format(
                    iteration, ARGS.iterations
                ))
                continue

            valid_fitness = metrics["fitness"].masked_fill(~valid_candidates, -float("inf"))
            elite_indices = torch.topk(
                valid_fitness, min(elite_count, valid_count), largest=True
            ).indices
            elite_base = base[elite_indices]
            new_mean = elite_base.mean(dim=0)
            new_std = elite_base.std(dim=0, unbiased=False)
            mean = (1.0 - ARGS.update_rate) * mean + ARGS.update_rate * new_mean
            std = (1.0 - ARGS.update_rate) * std + ARGS.update_rate * new_std
            std = std.clamp(ARGS.min_std, ARGS.max_std)

            top_index = int(torch.argmax(valid_fitness))
            score = float(metrics["fitness"][top_index].detach().cpu())
            if global_best is None or score > global_best["fitness"]:
                global_best = {
                    "fitness": score,
                    "iteration": iteration,
                    "candidate": top_index,
                    "success": bool(metrics["valid_success"][top_index].detach().cpu()),
                }
            print(
                "iter {:03d}/{:03d}: valid={}/{} best={:.4f} global={:.4f} success={} collision={}".format(
                    iteration, ARGS.iterations, valid_count, ARGS.population_size,
                    score, global_best["fitness"],
                    bool(metrics["valid_success"][top_index].detach().cpu()),
                    bool(metrics["collision"][top_index].detach().cpu()),
                )
            )

        metadata = {
            "schema_version": 2,
            "seed": ARGS.seed,
            "selected_object_index": int(search.ARGS.object_index),
            "object_label": label,
            "object_pose_world_xyzw": object_pose_np.tolist(),
            "dataset": str(Path(ARGS.dataset)),
            "ae_checkpoint": str(Path(ARGS.ae_checkpoint)),
            "flow_checkpoint": str(Path(ARGS.flow_checkpoint)),
            "hand": cfg.task.hand_config.name,
            "population_size": ARGS.population_size,
            "iterations": ARGS.iterations,
            "frames_per_trajectory": frame_count,
            "trajectory_steps_including_lift": sequence_steps,
            "hold_steps": ARGS.hold_steps,
            "trajectory_audit": {
                "max_object_point_abs_m": ARGS.max_object_point_abs,
                "max_hand_point_abs_m": ARGS.max_hand_point_abs,
                "max_same_point_step_m": ARGS.max_point_step,
                "iterations": iteration_audits,
            },
            "camera_width": ARGS.width,
            "camera_height": ARGS.height,
            "camera_intrinsics": intrinsics.tolist(),
            "camera_mount": "head_link",
            "camera_position_head_link": list(ARGS.camera_position),
            "camera_rpy_head_link_degrees": list(ARGS.camera_rpy_deg),
            "depth": "raw Isaac Gym IMAGE_DEPTH, negative meters; valid range from --near/--far",
            "point_cloud_frame": "object_points_base and hand_points_base use robot_base_link; object_points_camera is camera-CV diagnostic data",
            "robot_base_link": robot_base_link,
            "camera_pose_frame": "Isaac Gym camera frame in world, position plus quaternion xyzw",
            "camera_cv_to_isaac_xyz": "[z, -x, -y]",
            "object_segmentation_id": 2,
            "hand_segmentation_id": 3,
            "hand_body_names": hand_body_names,
            "object_mask": "visible object pixels from IMAGE_SEGMENTATION == 2",
            "object_points_camera": "fixed-size sampled visible points deprojected from depth and object_mask, camera CV right/down/forward",
            "object_points_base": "complete simulator asset cloud transformed from object-local through world into the robot base link frame at each frame",
            "hand_points_base": "fixed-size Wuji visual-mesh surface samples transformed from hand links through world into the robot base link frame at each frame",
            "robot_base_pose_world": "robot base link pose in world per frame, position plus quaternion xyzw",
            "point_counts_columns": [
                "valid object depth pixels", "valid visible hand depth pixels",
                "modeled hand mesh points",
            ],
            "hand_mesh_urdf": str(hand_urdf_path.resolve()),
            "hand_mesh_links": [entry[0] for entry in hand_mesh_points],
            "hand_mesh_point_count": ARGS.num_points,
            "hand_mask": "visible active right-hand body pixels from IMAGE_SEGMENTATION == 3; may be empty when the hand is outside the view",
            "robot_qpos_qvel": "all robot DoFs in robot_dof_names order, radians/meters as defined by URDF",
            "frame_phase_values": {"0": "latent trajectory and post-lift", "1": "hold"},
            "best_candidate": global_best,
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print("Saved collection metadata to {}".format(output_dir / "metadata.json"))
    finally:
        if env is not None:
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
            del env
            import gc
            gc.collect()
            vec_task.EXISTING_SIM = None


if __name__ == "__main__":
    main()
