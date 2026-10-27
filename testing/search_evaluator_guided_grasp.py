"""Search a grasp trajectory with evaluator-ranked CEM, then replay the best.

Each generation samples a latent population, transforms decoded trajectories
and the object point cloud into the robot-base frame for evaluator scoring,
and updates a CEM distribution from the top candidates. A fraction of each
population explores noisy world-frame mutations of the best trajectory, which
are encoded back into latent space. Isaac Gym plays the selected trajectory
after search using the original environment-frame action.

Example::

    python -m testing.search_evaluator_guided_grasp \
      --dataset datasets/inspire_grasp.npz \
      --ae-checkpoint runs/simulation_guided_grasp/ae_simulation_guided.pt \
      --flow-checkpoint runs/simulation_guided_grasp/flow_simulation_guided.pt \
      --evaluator-checkpoint runs/simulation_guided_grasp/evaluator.pt \
      --object-index 3 --population-size 64 --iterations 20 \
      --action-noise-std 0.01 --action-noise-fraction 0.25 \
      task=grasp hand=g1_upper_right_wuji num_envs=1 \
      headless=False task.env.armController=pose
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--ae-checkpoint",
        default="runs/simulation_guided_grasp/ae_simulation_guided.pt",
    )
    parser.add_argument(
        "--flow-checkpoint",
        default="runs/simulation_guided_grasp/flow_simulation_guided.pt",
    )
    parser.add_argument(
        "--evaluator-checkpoint",
        default="runs/simulation_guided_grasp/evaluator.pt",
    )
    parser.add_argument(
        "--object-index", type=int, default=0,
        help="row index in the dataset; its object cloud and orientation are used",
    )
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--min-std", type=float, default=0.05)
    parser.add_argument("--max-std", type=float, default=3.0)
    parser.add_argument("--update-rate", type=float, default=0.7)
    parser.add_argument("--action-noise-std", type=float, default=0.01)
    parser.add_argument("--action-noise-fraction", type=float, default=0.25)
    parser.add_argument("--max-base-abs", type=float, default=4.0)
    parser.add_argument("--post-lift-height", type=float, default=0.10)
    parser.add_argument("--post-lift-steps", type=int, default=10)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument(
        "--no-randomize-object-position", action="store_true",
        help="use the dataset object position; orientation is always kept from the dataset",
    )
    parser.add_argument("--seed", type=int, default=42)
    args, hydra_args = parser.parse_known_args()
    return args, hydra_args


def _validate_args(args: argparse.Namespace) -> None:
    if args.population_size < 1 or args.iterations < 1:
        raise ValueError("population-size and iterations must be positive")
    if not 0.0 < args.elite_fraction <= 1.0:
        raise ValueError("elite-fraction must be in (0, 1]")
    if args.initial_std <= 0 or args.min_std <= 0 or args.max_std < args.min_std:
        raise ValueError("invalid CEM standard-deviation bounds")
    if not 0.0 < args.update_rate <= 1.0:
        raise ValueError("update-rate must be in (0, 1]")
    if args.action_noise_std < 0 or not 0.0 <= args.action_noise_fraction <= 1.0:
        raise ValueError("invalid action-noise settings")
    if args.max_base_abs <= 0:
        raise ValueError("max-base-abs must be positive")
    if args.post_lift_steps < 0 or args.post_lift_height < 0 or args.hold_steps < 0:
        raise ValueError("post-lift and hold settings must be non-negative")


def _validate_checkpoint_weights(checkpoint, checkpoint_name: str, torch) -> None:
    invalid = [
        name
        for name, value in checkpoint["model"].items()
        if torch.is_tensor(value)
        and torch.is_floating_point(value)
        and not torch.isfinite(value).all()
    ]
    if invalid:
        raise ValueError(
            f"{checkpoint_name} contains non-finite model weights, including "
            f"{invalid[0]}; regenerate this checkpoint before searching"
        )


def _cem_search(
    args,
    ae,
    flow,
    evaluator,
    latent_shape,
    latent_dim,
    local_cloud,
    object_pose,
    base_pose,
    eef_offset,
    world_trajectory_fn,
    world_pointcloud_fn,
    world_to_base_trajectory_fn,
    world_to_base_pointcloud_fn,
    object_trajectory_fn,
    smooth_trajectory_noise_fn,
):
    import torch

    world_cloud = world_pointcloud_fn(local_cloud, object_pose)
    base_cloud = world_to_base_pointcloud_fn(world_cloud, base_pose)
    mean = torch.zeros(latent_dim, dtype=torch.float32, device=object_pose.device)
    std = torch.full_like(mean, float(args.initial_std))
    best = None
    parent_base = None
    parent_decoded = None
    parent_world = None
    parent_fitness = float("-inf")
    elite_count = max(1, int(np.ceil(args.population_size * args.elite_fraction)))
    exploration_count = 0
    if args.action_noise_std > 0 and args.action_noise_fraction > 0 and args.population_size > 1:
        exploration_count = min(
            args.population_size - 1,
            int(round(args.population_size * args.action_noise_fraction)),
        )
        exploration_count = max(1, exploration_count)

    for iteration in range(1, args.iterations + 1):
        candidate_base = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            args.population_size, latent_dim, device=object_pose.device
        )
        candidate_base.clamp_(-args.max_base_abs, args.max_base_abs)
        candidate_base[0] = mean if parent_base is None else parent_base
        with torch.no_grad():
            latent = flow.inverse(candidate_base).reshape(
                args.population_size, *latent_shape
            )
            candidate_decoded = ae.decode(latent)
            pose_batch = object_pose.expand(args.population_size, -1)
            candidate_world = world_trajectory_fn(
                candidate_decoded, pose_batch, eef_offset
            )
            if parent_decoded is not None:
                candidate_decoded[0] = parent_decoded
                candidate_world[0] = parent_world

            if exploration_count:
                exploration_parent = (
                    parent_world.unsqueeze(0)
                    if parent_world is not None else candidate_world[:1]
                )
                noisy_world = smooth_trajectory_noise_fn(
                    exploration_parent.expand(exploration_count, -1, -1).clone(),
                    args.action_noise_std,
                )
                noisy_world[..., 3:7] = torch.nn.functional.normalize(
                    noisy_world[..., 3:7], dim=-1
                )
                noisy_decoded = object_trajectory_fn(
                    noisy_world, pose_batch[:exploration_count], eef_offset
                )
                noisy_latent = ae.encode(noisy_decoded)
                noisy_base, _ = flow(noisy_latent.reshape(exploration_count, -1))
                candidate_base[-exploration_count:] = noisy_base.clamp(
                    -args.max_base_abs, args.max_base_abs
                )
                candidate_decoded[-exploration_count:] = noisy_decoded
                candidate_world[-exploration_count:] = noisy_world

            candidate_base_pose = base_pose.expand(args.population_size, -1)
            evaluator_trajectory = world_to_base_trajectory_fn(
                candidate_world, candidate_base_pose
            )
            scores = evaluator(
                base_cloud.expand(args.population_size, -1, -1),
                evaluator_trajectory,
            ).reshape(-1)
            if not torch.isfinite(scores).all():
                raise FloatingPointError(
                    "evaluator produced a non-finite score; check its checkpoint and inputs"
                )
            fitness = scores

            elite_indices = torch.topk(fitness, elite_count, largest=True).indices
            elite_base = candidate_base[elite_indices]
            new_mean = elite_base.mean(dim=0)
            new_std = elite_base.std(dim=0, unbiased=False)
            mean = (1.0 - args.update_rate) * mean + args.update_rate * new_mean
            std = (1.0 - args.update_rate) * std + args.update_rate * new_std
            std.clamp_(args.min_std, args.max_std)

            top_index = int(fitness.argmax().item())
            top_fitness = float(fitness[top_index].item())
            if top_fitness > parent_fitness:
                parent_fitness = top_fitness
                parent_base = candidate_base[top_index].clone()
                parent_decoded = candidate_decoded[top_index].clone()
                parent_world = candidate_world[top_index].clone()
                best = {
                    "score": float(scores[top_index].item()),
                    "fitness": top_fitness,
                    "base": parent_base.clone(),
                    "decoded": parent_decoded.clone(),
                    "iteration": iteration,
                }

        print(
            f"iter {iteration:03d}/{args.iterations}: "
            f"best_score={float(scores.max()):.4f} "
            f"global_score={best['score']:.4f} "
            f"batch_mean={float(scores.mean()):.4f} "
            f"std={float(std.mean()):.4f} "
            f"world_noise={exploration_count}/{args.population_size}"
        )

    if best is None:
        raise RuntimeError("CEM did not produce a candidate")
    return best


def _render_viewer(env) -> bool:
    gym = env.gym
    if env.viewer is None or gym.query_viewer_has_closed(env.viewer):
        return False
    if str(env.device) != "cpu":
        gym.fetch_results(env.sim, True)
    gym.step_graphics(env.sim)
    gym.draw_viewer(env.viewer, env.sim, True)
    gym.sync_frame_time(env.sim)
    return not gym.query_viewer_has_closed(env.viewer)


def _replay(env, best, object_pose, eef_offset, args, world_trajectory_fn,
            append_lift_fn, hand_action_fn):
    import torch

    if env.viewer is None:
        raise RuntimeError("Isaac Gym did not create a viewer; run with headless=False")
    if getattr(env, "dual_arm", False):
        env.active_arm_ids.fill_(False)  # False selects the configured right arm.

    env_ids = torch.zeros(1, dtype=torch.long, device=env.device)
    env.reset_idx(env_ids, object_init_pose=object_pose.detach().cpu().numpy()[None])
    actual_pose = env.object_init_states[0, :7].detach().clone()
    initial_object_z = actual_pose[2].clone()
    world_action = world_trajectory_fn(
        best["decoded"].to(env.device).unsqueeze(0),
        actual_pose.unsqueeze(0),
        eef_offset,
    )
    world_action = append_lift_fn(world_action, args.post_lift_steps, args.post_lift_height)
    max_lift = torch.zeros((), dtype=torch.float32, device=env.device)
    success_seen = False
    collision_seen = False

    print(
        f"Playing CEM-best trajectory: predicted_score={best['score']:.4f}, "
        f"search_iteration={best['iteration']}, "
        f"object_pose={actual_pose.cpu().tolist()}"
    )
    with torch.no_grad():
        final_action = env.no_op_action.clone()
        for step in range(world_action.shape[1] + args.hold_steps):
            if not _render_viewer(env):
                break
            action_index = min(step, world_action.shape[1] - 1)
            action = env.no_op_action.clone()
            action[:, :7] = world_action[:, action_index, :7]
            action[:, env.hand_dof_start_idx:] = hand_action_fn(
                env, world_action[:, action_index, 7:]
            )
            final_action.copy_(action)
            env.step(action)
            lift = env.root_state_tensor[env.object_indices[0], 2] - initial_object_z
            max_lift = torch.maximum(max_lift, lift)
            success_seen = success_seen or bool(env.successes[0].item())
            collision_seen = collision_seen or bool(env.has_hit_table[0].item())

        print(
            "Playback observation (not used for search): "
            f"success={success_seen}, collision={collision_seen}, "
            f"peak_object_lift={float(max_lift):.4f}m, "
            f"final_object_lift={float(env.root_state_tensor[env.object_indices[0], 2] - initial_object_z):.4f}m"
        )
        while _render_viewer(env):
            env.step(final_action)


def _run(cfg, args: argparse.Namespace) -> None:
    import torch
    import isaacgymenvs
    from isaacgymenvs.tasks.base import vec_task
    from isaacgymenvs.utils.utils import set_seed
    import tasks  # noqa: F401

    from memory.train_simulation_guided_grasp import (
        GraspEvaluator,
        _append_lift,
        _hand_action,
        _load_object_data,
        _object_trajectory,
        _smooth_trajectory_noise,
        _world_pointcloud,
        _world_trajectory,
        _world_to_base_pointcloud,
        _world_to_base_trajectory,
        _write_object_pool,
    )
    from memory.train_latent_flow import _make_flow
    from memory.train_trajectory_ae import _build_model

    _validate_args(args)
    random_state = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    cfg.seed = set_seed(args.seed, torch_deterministic=cfg.torch_deterministic)

    object_data = _load_object_data(args.dataset)
    if not 0 <= args.object_index < len(object_data["object_label"]):
        raise IndexError(
            f"--object-index must be in [0, {len(object_data['object_label'])})"
        )
    label = str(object_data["object_label"][args.object_index])
    local_cloud = torch.as_tensor(
        object_data["object_pointcloud"][args.object_index:args.object_index + 1].copy(),
        dtype=torch.float32,
    )
    object_pose_np = object_data["object_pose_world"][args.object_index].copy()
    quaternion_norm = np.linalg.norm(object_pose_np[3:7])
    if quaternion_norm < 1e-8:
        object_pose_np[3:7] = np.asarray([0, 0, 0, 1], dtype=np.float32)
    else:
        object_pose_np[3:7] /= quaternion_norm

    if not args.no_randomize_object_position:
        hand_cfg = cfg.task.hand_config
        bounds = np.asarray(
            hand_cfg.get("objectResetPositionRange", cfg.task.env.resetPositionRange),
            dtype=np.float32,
        )
        if bounds.shape != (3, 2):
            raise ValueError(f"expected object position bounds [3, 2], got {bounds.shape}")
        object_pose_np[:3] = random_state.uniform(
            bounds[:, 0], bounds[:, 1]
        ).astype(np.float32)
        object_pose_np[2] += float(np.mean(cfg.task.env.tableHeightRange))
    object_pose = torch.as_tensor(object_pose_np[None], dtype=torch.float32)
    print(
        f"Object {args.object_index}: {label}; "
        f"position_randomized={not args.no_randomize_object_position}, "
        "dataset orientation retained."
    )

    asset_list_path = _write_object_pool([label])
    cfg.task.env.asset.multiObject = True
    cfg.task.env.asset.multiObjectList = str(asset_list_path.relative_to("assets"))
    cfg.task.env.asset.randomizeObjectAssets = False
    if cfg.task.env.armController != "pose":
        raise ValueError("evaluator-guided grasp replay requires task.env.armController=pose")

    ae_checkpoint = torch.load(args.ae_checkpoint, map_location="cpu")
    flow_checkpoint = torch.load(args.flow_checkpoint, map_location="cpu")
    evaluator_checkpoint = torch.load(args.evaluator_checkpoint, map_location="cpu")
    for name, checkpoint in (
        ("AE checkpoint", ae_checkpoint),
        ("Flow checkpoint", flow_checkpoint),
        ("Evaluator checkpoint", evaluator_checkpoint),
    ):
        _validate_checkpoint_weights(checkpoint, name, torch)
    ae_config = dict(ae_checkpoint["config"])
    flow_config = dict(flow_checkpoint["config"])
    evaluator_config = dict(evaluator_checkpoint["config"])
    if evaluator_config.get("coordinate_frame") != "robot_base":
        raise ValueError(
            "the evaluator checkpoint must use coordinate_frame='robot_base'; "
            "retrain the evaluator with the updated training script"
        )
    if evaluator_config.get("architecture") != "pointnet_ae_fitness_v1":
        raise ValueError(
            "the evaluator checkpoint does not contain the point-cloud AE architecture; "
            "retrain it with the updated training script"
        )
    sequence_length = int(ae_config["sequence_length"])
    input_dim = int(ae_config["input_dim"])
    hand_dim = input_dim - 9
    if sequence_length != 20:
        raise ValueError(f"expected a 20-frame AE checkpoint, got {sequence_length}")
    latent_shape = tuple(int(value) for value in flow_config["latent_shape"])
    expected_latent_shape = (int(ae_config["num_tokens"]), int(ae_config["latent_dim"]))
    latent_dim = int(flow_config["latent_dim"])
    if latent_shape != expected_latent_shape or latent_dim != int(np.prod(latent_shape)):
        raise ValueError("AE and Flow checkpoint latent shapes do not match")
    expected_action_dim = 7 + hand_dim
    if int(evaluator_config.get("action_dim", expected_action_dim)) != expected_action_dim:
        raise ValueError(
            f"evaluator expects action_dim={evaluator_config['action_dim']}, "
            f"but the AE decodes {expected_action_dim} action channels"
        )

    dual_arm = bool(cfg.task.hand_config.get(
        "dualArm", cfg.task.hand_config.get("dualArmInspire", False)
    ))
    if dual_arm:
        cfg.task.env.activeArm = 1
        offset_value = cfg.task.hand_config.trajectoryEefOffsetByArm.right
    else:
        offset_value = cfg.task.hand_config.get("trajectoryEefOffset", None)
    device = torch.device(cfg.sim_device)
    eef_offset = (
        None if offset_value is None
        else torch.as_tensor(offset_value, dtype=torch.float32, device=device)
    )
    if eef_offset is not None and eef_offset.shape != (7,):
        raise ValueError("trajectory EEF offset must contain xyz+xyzw")

    base_pose_value = cfg.task.hand_config.get(
        "robotStartPose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    base_pose = torch.as_tensor(
        base_pose_value, dtype=torch.float32, device=device
    ).reshape(1, 7)
    base_pose[:, 3:7] = torch.nn.functional.normalize(base_pose[:, 3:7], dim=-1)

    ae = _build_model(ae_config).to(device)
    ae.load_state_dict(ae_checkpoint["model"])
    flow = _make_flow(flow_config, device)
    flow.load_state_dict(flow_checkpoint["model"])
    evaluator = GraspEvaluator(
        action_dim=expected_action_dim,
        point_latent_dim=int(evaluator_config["point_latent_dim"]),
        reconstruction_points=int(evaluator_config["reconstruction_points"]),
    ).to(device)
    evaluator.load_state_dict(evaluator_checkpoint["model"])
    evaluator.set_fitness_stats(
        evaluator_config.get("fitness_mean", 0.0),
        evaluator_config.get("fitness_std", 1.0),
    )
    for model in (ae, flow, evaluator):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    object_pose = object_pose.to(device)
    local_cloud = local_cloud.to(device)
    print(
        f"Evaluator-only search: population={args.population_size}, "
        f"CEM iterations={args.iterations}, "
        f"action_noise_fraction={args.action_noise_fraction:.2f}; "
        "no candidate is sent to Isaac Gym during search."
    )
    best = _cem_search(
        args, ae, flow, evaluator, latent_shape, latent_dim,
        local_cloud, object_pose, base_pose, eef_offset, _world_trajectory,
        _world_pointcloud, _world_to_base_trajectory,
        _world_to_base_pointcloud, _object_trajectory,
        _smooth_trajectory_noise,
    )
    print(
        f"Search complete: best predicted score={best['score']:.4f}; "
        f"candidate from CEM iteration={best['iteration']}."
    )

    cfg.num_envs = 1
    cfg.task.env.numEnvs = 1
    cfg.task.env.episodeLength = max(
        100000, sequence_length + args.post_lift_steps + args.hold_steps + 2
    )
    cfg.task.env.randomizeTrackingReference = False
    cfg.task.env.randomizeGraspPose = False
    cfg.task.env.resetDofPosRandomInterval = 0.0
    cfg.task.env.resetHandDofPosFullRange = False
    cfg.task.env.render.enable = False
    cfg.force_render = True
    cfg.headless = False
    env = None
    try:
        env = isaacgymenvs.make(
            cfg.seed, cfg.task_name, 1,
            cfg.sim_device, cfg.rl_device, cfg.graphics_device_id,
            False, cfg.multi_gpu, False, cfg.force_render, cfg,
        )
        if str(env.device) != str(device):
            raise ValueError(f"model device {device} does not match simulator device {env.device}")
        _replay(
            env, best, object_pose[0], eef_offset, args,
            _world_trajectory, _append_lift, _hand_action,
        )
    finally:
        if env is not None:
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(torch.device(env.device))
            del env
            vec_task.EXISTING_SIM = None


def main() -> None:
    args, hydra_args = _parse_args()
    sys.argv = [sys.argv[0], *hydra_args]
    # Isaac Gym must be imported before torch, matching the training entry point.
    import isaacgym  # noqa: F401
    import hydra

    @hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
    def hydra_entry(cfg) -> None:
        _run(cfg, args)

    hydra_entry()


if __name__ == "__main__":
    main()
