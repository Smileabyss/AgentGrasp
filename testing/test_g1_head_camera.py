"""Preview an adjustable camera attached to the G1 head in the Grasp scene.

Example::

    python -m testing.test_g1_head_camera \
        --camera-position 0.1 0.0 0.45 \
        --camera-rpy-deg 0 45 0 \
        task=grasp hand=g1_upper_right_wuji num_envs=1 headless=False \
        task.env.armController=pose task.env.enablePointCloud=False

Camera position is in ``head_link`` coordinates. RPY is roll, pitch, yaw in
degrees. The preview enables the Grasp task's realistic table/wall and texture
randomization so the RGB camera shows the same rendered scene used in training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        default=(0.1, 0.0, 0.45),
        metavar=("X", "Y", "Z"),
        help="camera translation in head_link coordinates, in meters",
    )
    parser.add_argument(
        "--camera-rpy-deg",
        type=float,
        nargs=3,
        default=(0.0, 45.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="camera rotation in head_link coordinates, in degrees",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=5.0)
    parser.add_argument(
        "--record-path",
        default=None,
        help="save one rendered RGB/depth frame and scene poses to this directory",
    )
    args, hydra_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0], *hydra_argv]
    return args


ARGS = _parse_args()


def _configure_camera(cfg: DictConfig, gymapi) -> None:
    cfg.num_envs = 1
    cfg.task.env.numEnvs = 1
    cfg.task.env.render.enable = True
    cfg.task.env.render.camera_ids = [1]
    cfg.task.env.render.data_type = "rgb,depth"
    cfg.task.env.render.appearance_realistic = True
    cfg.task.env.render.randomize = True
    cfg.task.env.render.randomization_params.camera_pos = [-0.02, 0.02]
    cfg.task.env.render.randomization_params.camera_quat = [-0.02, 0.02]
    cfg.task.env.render.randomization_params.table_xyz = [0.05, 0.02, 0.0]
    cfg.task.env.enablePointCloud = ARGS.record_path is not None
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.task.env.randomEpisodeLength = False
    cfg.task.env.resetRandomRot = "fixed"
    cfg.task.env.randomizeGraspPose = False

    if cfg.task.hand_config.get("dualArmInspire", False):
        cfg.task.env.activeArm = 1  # G1 right arm

    cam = cfg.task.env.camera_config.camera_1
    cam.mount = "head_link"
    cam.width = ARGS.width
    cam.height = ARGS.height
    cam.depth_range = [ARGS.near, ARGS.far]
    focal_length = ARGS.width / (2.0 * math.tan(math.radians(ARGS.fov_deg) / 2.0))
    cam.intrinsics = [
        [focal_length, 0.0, ARGS.width / 2.0],
        [0.0, focal_length, ARGS.height / 2.0],
        [0.0, 0.0, 1.0],
    ]

    roll, pitch, yaw = (math.radians(value) for value in ARGS.camera_rpy_deg)
    quat = gymapi.Quat.from_euler_zyx(yaw, pitch, roll)
    OmegaConf.update(
        cfg,
        "task.env.camera_config.camera_1.position",
        list(ARGS.camera_position),
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "task.env.camera_config.camera_1.quaternion",
        [quat.x, quat.y, quat.z, quat.w],
        force_add=True,
    )


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    near, far = ARGS.near, ARGS.far
    distance = -depth
    valid = np.isfinite(distance) & (distance >= near) & (distance <= far)
    normalized = np.zeros(distance.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(distance[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        scaled = np.clip((distance[valid] - low) / (high - low), 0.0, 1.0)
        normalized[valid] = (255.0 * (1.0 - scaled)).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[~valid] = 0
    return preview


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Transform object-local XYZ points by an Isaac Gym xyzw pose."""
    x, y, z, w = pose[3:]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return points @ rotation.T + pose[:3]


def _world_to_camera(points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    """Transform world XYZ points into the CV camera frame (right, down, forward)."""
    x, y, z, w = camera_pose[3:]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    points_isaac = (points - camera_pose[:3]) @ rotation
    # Isaac Gym's camera frame is forward-left-up; camera intrinsics use the
    # conventional computer-vision frame right-down-forward.
    return np.stack(
        [-points_isaac[:, 1], -points_isaac[:, 2], points_isaac[:, 0]], axis=-1
    )


def _record_frame(env, camera_handle, rgba: np.ndarray, depth: np.ndarray) -> None:
    if ARGS.record_path is None:
        return
    output_dir = Path(ARGS.record_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_transform = env.gym.get_camera_transform(env.sim, env.envs[0], camera_handle)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    camera_pose = np.array(
        [
            camera_transform.p.x,
            camera_transform.p.y,
            camera_transform.p.z,
            camera_transform.r.x,
            camera_transform.r.y,
            camera_transform.r.z,
            camera_transform.r.w,
        ],
        dtype=np.float32,
    )
    object_pose = env.root_state_tensor[env.object_indices[0], :7].detach().cpu().numpy().astype(np.float32)
    object_points_local = np.asarray(env.object_pcls[0], dtype=np.float32)
    object_points_world = _transform_points(object_points_local, object_pose)
    object_points_camera = _world_to_camera(object_points_world, camera_pose)
    output_file = output_dir / "frame_000000.npz"
    camera_intrinsics = np.asarray(
        env.camera_cfg["camera_1"]["intrinsics"], dtype=np.float32
    )
    with (output_dir / "camera_intrinsics.json").open("w") as stream:
        json.dump(
            {
                "matrix": camera_intrinsics.tolist(),
                "width": ARGS.width,
                "height": ARGS.height,
                "horizontal_fov_deg": ARGS.fov_deg,
            },
            stream,
            indent=2,
        )
    cv2.imwrite(
        str(output_dir / "frame_000000_rgb.png"),
        cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR),
    )
    # Preserve Isaac Gym's raw depth tensor exactly; it stores negative view
    # distances. The NPZ also includes this same unconverted array.
    np.save(output_dir / "frame_000000_depth.npy", depth)
    np.savez_compressed(
        output_file,
        rgb=rgba,
        depth=depth,
        camera_pose_world=camera_pose,
        object_pose_world=object_pose,
        object_points_local=object_points_local,
        object_points_world=object_points_world,
        object_points_camera=object_points_camera,
        camera_intrinsics=camera_intrinsics,
    )
    print(f"Recorded camera frame and scene data to {output_file}")


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    from isaacgym import gymapi
    import torch
    import isaacgymenvs
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import tasks  # noqa: F401  # registers the Grasp task

    if cfg.headless:
        raise ValueError("Camera preview requires headless=False")

    _configure_camera(cfg, gymapi)
    set_np_formatting()
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    env = isaacgymenvs.make(
        cfg.seed,
        cfg.task_name,
        1,
        cfg.sim_device,
        cfg.rl_device,
        cfg.graphics_device_id,
        False,
        cfg.multi_gpu,
        False,
        cfg.force_render,
        cfg,
    )

    gym = env.gym
    try:
        env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
        env.reset_idx(env_ids)

        viewer = env.viewer
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(1.6, -2.4, 1.5),
            gymapi.Vec3(0.1, 0.1, 0.9),
        )
        print(
            "Head camera: position={} m, RPY={} deg, size={}x{}, FOV={} deg. "
            "Press q in either image window to quit.".format(
                tuple(ARGS.camera_position),
                tuple(ARGS.camera_rpy_deg),
                ARGS.width,
                ARGS.height,
                ARGS.fov_deg,
            )
        )

        rgb_tensor = env.camera_rgb_tensor_lists[0][0]
        depth_tensor = env.camera_depth_tensor_lists[0][0]
        recorded = False
        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(env.sim)
            gym.fetch_results(env.sim, True)
            gym.step_graphics(env.sim)
            gym.render_all_camera_sensors(env.sim)
            gym.start_access_image_tensors(env.sim)
            try:
                rgba = rgb_tensor.cpu().numpy().copy()
                depth = depth_tensor.cpu().numpy().copy()
            finally:
                gym.end_access_image_tensors(env.sim)

            if not recorded:
                _record_frame(env, env.camera_handles[0], rgba, depth)
                recorded = True

            rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
            cv2.imshow("G1 head camera | RGB", rgb)
            cv2.imshow("G1 head camera | depth (autoscaled)", _depth_preview(depth))
            gym.draw_viewer(viewer, env.sim, True)
            gym.sync_frame_time(env.sim)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        if env.viewer is not None:
            gym.destroy_viewer(env.viewer)
        gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
