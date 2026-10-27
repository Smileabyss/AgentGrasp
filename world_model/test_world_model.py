from __future__ import annotations

import importlib
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Allow direct execution via ``python world_model/test_world_model.py``.
PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import gym
import hydra
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from omegaconf import DictConfig, OmegaConf, open_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import tasks
import torch

from world_model.policy_runtime import _apply_checkpoint_compat_overrides, _build_policy_runner, _resolve_checkpoint_path
from world_model.PPModel import PPModel
from world_model.algo.pn_utils.world_model import (
    build_trajectory_world_model,
    chamfer_distance,
)
from world_model.train_world_model import _select_contact_points_from_hand_points


class WorldModelViewer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8081,
        object_point_size: float = 0.004,
        contact_point_size: float = 0.005,
        sleep_dt: float = 0.05,
        show_contact: bool = True,
    ):
        try:
            viser = importlib.import_module("viser")
        except ImportError as exc:
            raise ImportError("test_world_model.py requires the 'viser' package for visualization.") from exc

        self.server = viser.ViserServer(host=host, port=port)
        self.object_point_size = float(object_point_size)
        self.contact_point_size = float(contact_point_size)
        self.sleep_dt = float(sleep_dt)
        self.show_contact = bool(show_contact)
        self._trajectory_handles = None

    def wait_for_client(self) -> None:
        print("Open the Viser URL in a browser; trajectory playback will start after it connects.")
        while not self.server.get_clients():
            time.sleep(0.1)
        print("Viser client connected; starting trajectory playback.")

    def update(
        self,
        *,
        env_index: int,
        current_object_points: np.ndarray,
        actual_next_object_points: np.ndarray,
        predicted_next_object_points: np.ndarray,
        current_contact_points: np.ndarray | None,
        ordered_l2_error: float,
        chamfer_error: float,
    ) -> None:
        prefix = f"world_model/env_{env_index}"
        self.server.scene.add_point_cloud(
            f"{prefix}/current_object",
            current_object_points,
            point_size=self.object_point_size * 0.75,
            point_shape="circle",
            colors=(160, 160, 160),
        )
        self.server.scene.add_point_cloud(
            f"{prefix}/actual_next_object",
            actual_next_object_points,
            point_size=self.object_point_size,
            point_shape="circle",
            colors=(0, 120, 255),
        )
        self.server.scene.add_point_cloud(
            f"{prefix}/predicted_next_object",
            predicted_next_object_points,
            point_size=self.object_point_size,
            point_shape="circle",
            colors=(255, 140, 0),
        )
        if self.show_contact and current_contact_points is not None:
            self.server.scene.add_point_cloud(
                f"{prefix}/current_contact",
                current_contact_points,
                point_size=self.contact_point_size,
                point_shape="circle",
                colors=(80, 220, 120),
            )
        print(
            f"viewer env={env_index} ordered_l2={ordered_l2_error:.6f} chamfer={chamfer_error:.6f}"
        )
        if self.sleep_dt > 0:
            time.sleep(self.sleep_dt)

    def update_trajectory_comparison(
        self,
        *,
        env_index: int,
        timestep: int,
        hand_points: np.ndarray,
        ground_truth_points: np.ndarray,
        predicted_points: np.ndarray,
        display_offset: np.ndarray,
        ordered_l2_error: float,
        chamfer_error: float,
    ) -> None:
        prefix = f"world_model/env_{env_index}"
        left_offset = display_offset * 0.5
        if self._trajectory_handles is None:
            self._trajectory_handles = {
                "ground_truth_hand": self.server.scene.add_point_cloud(
                    f"{prefix}/ground_truth_hand",
                    hand_points - left_offset,
                    point_size=self.contact_point_size,
                    point_shape="circle",
                    colors=(80, 220, 120),
                ),
                "ground_truth_object": self.server.scene.add_point_cloud(
                    f"{prefix}/ground_truth_trajectory",
                    ground_truth_points - left_offset,
                    point_size=self.object_point_size,
                    point_shape="circle",
                    colors=(0, 150, 255),
                ),
                "predicted_hand": self.server.scene.add_point_cloud(
                    f"{prefix}/predicted_hand",
                    hand_points + left_offset,
                    point_size=self.contact_point_size,
                    point_shape="circle",
                    colors=(80, 220, 120),
                ),
                "predicted_object": self.server.scene.add_point_cloud(
                    f"{prefix}/predicted_trajectory",
                    predicted_points + left_offset,
                    point_size=self.object_point_size,
                    point_shape="circle",
                    colors=(255, 140, 0),
                ),
            }
        else:
            self._trajectory_handles["ground_truth_hand"].points = hand_points - left_offset
            self._trajectory_handles["ground_truth_object"].points = ground_truth_points - left_offset
            self._trajectory_handles["predicted_hand"].points = hand_points + left_offset
            self._trajectory_handles["predicted_object"].points = predicted_points + left_offset
        print(
            f"playback frame={timestep} ordered_l2={ordered_l2_error:.6f} "
            f"chamfer={chamfer_error:.6f}"
        )
        if self.sleep_dt > 0:
            time.sleep(self.sleep_dt)


def _get_wm_cfg(cfg: DictConfig):
    default_cfg = OmegaConf.create(
        {
            "checkpoint": "",
            "dataset": None,
            "candidate_file": None,
            "num_random_candidates": 0,
            "random_seed": 0,
            "hold_viewer": True,
            "env_indices": "all",
            "num_rounds": None,
            "action_source": "policy",
            "viewer_host": "127.0.0.1",
            "viewer_port": 8081,
            "viewer_sleep_dt": 0.05,
            "enable_viewer": True,
            "show_contact": True,
            "object_point_size": 0.004,
            "contact_point_size": 0.005,
            "log_interval": 10,
            "world_model_device": "cuda" if torch.cuda.is_available() else "cpu",
            "preprocess_device": "cuda" if torch.cuda.is_available() else "cpu",
            "points_per_link": None,
            "num_contact_points": None,
            "outside_distance_threshold": None,
            "root_link_names": None,
            "urdf_path": None,
        }
    )
    if "wm" not in cfg or cfg.wm is None:
        return default_cfg
    return OmegaConf.merge(default_cfg, cfg.wm)


def _normalize_wm_env_indices(wm_cfg, num_envs: int) -> List[int]:
    raw_env_indices = wm_cfg.get("env_indices", None)
    if raw_env_indices is None:
        return list(range(num_envs))

    if isinstance(raw_env_indices, str):
        stripped = raw_env_indices.strip().lower()
        if stripped in {"", "all", "*"}:
            return list(range(num_envs))
        parts = [part.strip() for part in raw_env_indices.split(",") if part.strip()]
        env_indices = [int(part) for part in parts]
    else:
        env_indices = [int(idx) for idx in raw_env_indices]

    if not env_indices:
        raise ValueError("wm.env_indices must contain at least one environment index.")

    unique_env_indices = []
    seen = set()
    for env_index in env_indices:
        if env_index < 0 or env_index >= num_envs:
            raise ValueError(
                f"wm.env_indices requests env {env_index}, but the current simulation only has "
                f"{num_envs} env(s), so valid indices are [0, {num_envs - 1}]. "
                f"Increase num_envs or use wm.env_indices=all."
            )
        if env_index not in seen:
            unique_env_indices.append(env_index)
            seen.add(env_index)
    return unique_env_indices


def _infer_required_num_envs_from_wm_cfg(wm_cfg) -> int | None:
    raw_env_indices = wm_cfg.get("env_indices", None)
    if raw_env_indices is None:
        return None

    if isinstance(raw_env_indices, str):
        stripped = raw_env_indices.strip().lower()
        if stripped in {"", "all", "*"}:
            return None
        parts = [part.strip() for part in raw_env_indices.split(",") if part.strip()]
        env_indices = [int(part) for part in parts]
    else:
        env_indices = [int(idx) for idx in raw_env_indices]

    if not env_indices:
        return None
    return max(env_indices) + 1


def _parse_num_rounds(raw_value) -> int | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip().lower()
        if stripped in {"", "none", "null", "inf", "infinite"}:
            return None
        parsed_value = int(stripped)
    else:
        parsed_value = int(raw_value)
    return parsed_value if parsed_value > 0 else None


def _parse_optional_root_link_names(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        if "," in stripped:
            names = [part.strip() for part in stripped.split(",") if part.strip()]
            return names if len(names) > 1 else names[0]
        return stripped
    if isinstance(raw_value, (list, tuple)):
        names = [str(item).strip() for item in raw_value if str(item).strip()]
        if not names:
            return None
        return names if len(names) > 1 else names[0]
    return raw_value


def _safe_torch_load(path: str | Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_default_world_model_checkpoint() -> Path:
    candidate_paths = list((PROJECT_ROOT / "runs_world_model").glob("**/best.pt"))
    if not candidate_paths:
        candidate_paths = list((PROJECT_ROOT / "runs_world_model").glob("**/latest.pt"))
    if not candidate_paths:
        raise FileNotFoundError(
            "Cannot find a world model checkpoint under runs_world_model/. "
            "Please pass +wm.checkpoint=/path/to/best.pt"
        )
    return max(candidate_paths, key=lambda path: path.stat().st_mtime)


def _apply_policy_mode_defaults(cfg: DictConfig) -> None:
    with open_dict(cfg):
        cfg.task.env.observationType = "eefpose+objinitpose+objpcl"
        cfg.task.env.armController = "pose"
        cfg.task.env.enablePointCloud = True
        cfg.task.env.randomizeTrackingReference = True
        cfg.task.env.randomizeGraspPose = True
        if "train" not in cfg or cfg.train is None:
            cfg.train = OmegaConf.load(PROJECT_ROOT / "tasks" / "train" / "PPOOneStep.yaml")
        cfg.train.params.is_vision = True
        cfg.test = True


def _resolve_world_model_setup(cfg: DictConfig, wm_cfg) -> Dict[str, object]:
    checkpoint_path = (
        _resolve_checkpoint_path(wm_cfg.checkpoint, label="world model checkpoint")
        if wm_cfg.checkpoint
        else _find_default_world_model_checkpoint()
    )
    checkpoint = _safe_torch_load(checkpoint_path, map_location="cpu")
    if wm_cfg.dataset and checkpoint.get("point_cloud_frame") != "robot_base":
        raise ValueError(
            "Offline latent-search evaluation requires a checkpoint trained with "
            "robot-base-frame point clouds. Train a new checkpoint from scratch and pass it with +wm.checkpoint."
        )
    checkpoint_args = checkpoint.get("args", {})

    history_horizon = int(checkpoint_args["history_horizon"])
    future_horizon = int(checkpoint_args["future_horizon"])
    points_per_link = int(
        checkpoint_args["points_per_link"] if wm_cfg.points_per_link is None else wm_cfg.points_per_link
    )
    num_contact_points = int(
        checkpoint_args["num_contact_points"]
        if wm_cfg.num_contact_points is None
        else wm_cfg.num_contact_points
    )
    outside_distance_threshold = float(
        checkpoint_args["outside_distance_threshold"]
        if wm_cfg.outside_distance_threshold is None
        else wm_cfg.outside_distance_threshold
    )
    root_link_names = _parse_optional_root_link_names(
        checkpoint.get(
            "root_link_names",
            checkpoint_args.get("root_link_names", wm_cfg.root_link_names),
        )
        if wm_cfg.root_link_names is None
        else wm_cfg.root_link_names
    )
    urdf_path = (
        checkpoint_args.get("urdf_path", "assets/inspire_tac/fr3_inspire_tac_L_right_safety.urdf")
        if wm_cfg.urdf_path is None
        else wm_cfg.urdf_path
    )

    model = build_trajectory_world_model(
        object_point_dim=3,
        contact_point_dim=3,
        trajectory_feature_dim=int(checkpoint_args["trajectory_feature_dim"]),
        global_feature_dim=int(checkpoint_args["global_feature_dim"]),
        local_feature_dim=int(checkpoint_args["local_feature_dim"]),
        hidden_dim=int(checkpoint_args["hidden_dim"]),
        recurrent_layers=int(checkpoint_args["recurrent_layers"]),
    ).to(torch.device(wm_cfg.world_model_device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "history_horizon": history_horizon,
        "future_horizon": future_horizon,
        "points_per_link": points_per_link,
        "num_contact_points": num_contact_points,
        "outside_distance_threshold": outside_distance_threshold,
        "root_link_names": root_link_names,
        "urdf_path": urdf_path,
        "model": model,
    }


def _resolve_latent_search_candidate(dataset_path: str | Path, candidate_file: str | Path | None) -> Path:
    if candidate_file:
        path = Path(candidate_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Latent-search candidate not found: {path}")
        return path.resolve()

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Latent-search dataset directory not found: {path}")

    metadata_path = path / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        best = metadata.get("best_candidate") or {}
        if best.get("iteration") is not None and best.get("candidate") is not None:
            best_path = path / f"iteration_{int(best['iteration']):03d}" / (
                f"candidate_{int(best['candidate']):04d}.npz"
            )
            if best_path.is_file():
                return best_path.resolve()

    candidates = sorted(path.glob("iteration_*/candidate_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No candidate_*.npz files found under {path}")

    best_path = candidates[0]
    best_fitness = float("-inf")
    for candidate in candidates:
        with np.load(candidate, allow_pickle=False) as payload:
            fitness = float(np.asarray(payload["fitness"]).item()) if "fitness" in payload else float("-inf")
        if fitness > best_fitness:
            best_path, best_fitness = candidate, fitness
    return best_path.resolve()


def _resolve_latent_search_candidates(
    dataset_path: str | Path,
    candidate_file: str | Path | None,
    num_random_candidates: int,
    random_seed: int,
) -> List[Path]:
    if num_random_candidates < 0:
        raise ValueError("wm.num_random_candidates must be zero or greater.")
    if candidate_file:
        return [_resolve_latent_search_candidate(dataset_path, candidate_file)]

    dataset = Path(dataset_path).expanduser()
    if not dataset.is_absolute():
        dataset = Path.cwd() / dataset
    if dataset.is_file():
        if num_random_candidates > 1:
            raise ValueError("wm.num_random_candidates > 1 requires a dataset directory, not a single candidate file.")
        return [dataset.resolve()]

    if num_random_candidates == 0:
        return [_resolve_latent_search_candidate(dataset, None)]

    if not dataset.is_dir():
        raise FileNotFoundError(f"Latent-search dataset directory not found: {dataset}")
    candidates = sorted(dataset.glob("iteration_*/candidate_*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No candidate_*.npz files found under {dataset}")
    if num_random_candidates > len(candidates):
        raise ValueError(
            f"Requested {num_random_candidates} random candidates, but only {len(candidates)} are available under {dataset}."
        )

    indices = np.random.default_rng(int(random_seed)).choice(
        len(candidates), size=int(num_random_candidates), replace=False
    )
    selected = [candidates[int(index)].resolve() for index in indices]
    print(
        f"Randomly selected {len(selected)} of {len(candidates)} candidates "
        f"with seed={int(random_seed)}: {[path.name for path in selected]}"
    )
    return selected


def _latent_search_world_trajectories(candidate_path: Path):
    with np.load(candidate_path, allow_pickle=False) as payload:
        required = ("object_points_base", "hand_points_base", "robot_base_link")
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"{candidate_path} is missing required fields: {missing}")

        object_points = np.asarray(payload["object_points_base"], dtype=np.float32)
        hand_points = np.asarray(payload["hand_points_base"], dtype=np.float32)
        robot_base_link = str(np.asarray(payload["robot_base_link"]).item())
        actions = np.asarray(payload["actions"], dtype=np.float32) if "actions" in payload else None
        frame_phase = np.asarray(payload["frame_phase"]) if "frame_phase" in payload else None

    if not robot_base_link:
        raise ValueError(f"{candidate_path} has an empty robot_base_link frame label")

    return (
        torch.as_tensor(object_points, dtype=torch.float32),
        torch.as_tensor(hand_points, dtype=torch.float32),
        actions,
        frame_phase,
    )


def _rollout_world_model_from_initial_cloud(
    *,
    object_sequence: torch.Tensor,
    hand_points_sequence: torch.Tensor,
    world_model: torch.nn.Module,
    history_horizon: int,
    future_horizon: int,
    num_contact_points: int,
    outside_distance_threshold: float,
) -> torch.Tensor:
    if object_sequence.shape[0] < 2:
        raise ValueError("Need at least two recorded frames to evaluate a trajectory.")
    if hand_points_sequence.shape[0] != object_sequence.shape[0]:
        raise ValueError("Object and hand trajectories must have the same number of frames.")

    initial_object = object_sequence[0]
    initial_contact = _select_future_contact_points_for_rollout(
        hand_points=hand_points_sequence[:1],
        current_object_points=initial_object,
        num_contact_points=num_contact_points,
        outside_distance_threshold=outside_distance_threshold,
    )[0]
    object_history = [initial_object.clone() for _ in range(history_horizon)]
    contact_history = [initial_contact.clone() for _ in range(history_horizon)]
    predicted_states = [initial_object.clone()]
    device = next(world_model.parameters()).device

    next_timestep = 1
    while next_timestep < object_sequence.shape[0]:
        future_end = min(next_timestep + future_horizon, object_sequence.shape[0])
        future_contacts = _select_future_contact_points_for_rollout(
            hand_points=hand_points_sequence[next_timestep:future_end],
            current_object_points=predicted_states[-1],
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
        )
        chunk_len = int(future_contacts.shape[0])
        if chunk_len < future_horizon:
            future_contacts = torch.cat(
                [future_contacts, future_contacts[-1:].repeat(future_horizon - chunk_len, 1, 1)],
                dim=0,
            )

        with torch.no_grad():
            prediction = world_model(
                torch.stack(object_history[-history_horizon:]).unsqueeze(0).to(device),
                torch.stack(contact_history[-history_horizon:]).unsqueeze(0).to(device),
                future_contacts.unsqueeze(0).to(device),
            )

        predicted_chunk = prediction.object_future_points[0, :chunk_len].detach().cpu()
        for timestep in range(chunk_len):
            predicted_states.append(predicted_chunk[timestep].clone())
            object_history.append(predicted_chunk[timestep].clone())
            contact_history.append(future_contacts[timestep].clone())
        next_timestep += chunk_len

    return torch.stack(predicted_states, dim=0)


def run_offline_world_model_test(cfg: DictConfig) -> Dict[str, object]:
    wm_cfg = _get_wm_cfg(cfg)
    candidate_paths = _resolve_latent_search_candidates(
        wm_cfg.dataset,
        wm_cfg.candidate_file,
        int(wm_cfg.num_random_candidates),
        int(wm_cfg.random_seed),
    )
    wm_setup = _resolve_world_model_setup(cfg, wm_cfg)

    viewer = None
    if bool(wm_cfg.enable_viewer):
        viewer = WorldModelViewer(
            host=str(wm_cfg.viewer_host),
            port=int(wm_cfg.viewer_port),
            object_point_size=float(wm_cfg.object_point_size),
            contact_point_size=float(wm_cfg.contact_point_size),
            sleep_dt=float(wm_cfg.viewer_sleep_dt),
            show_contact=False,
        )
        print(f"viewer: http://{wm_cfg.viewer_host}:{int(wm_cfg.viewer_port)}")
        print("viewer colors: hand = green; ground truth = blue (left); prediction = orange (right)")
        viewer.wait_for_client()

    trajectories = []
    ordered_l2_errors = []
    chamfer_errors = []
    for candidate_index, candidate_path in enumerate(candidate_paths, start=1):
        object_sequence, hand_points_sequence, actions, frame_phase = _latent_search_world_trajectories(candidate_path)
        print(
            f"candidate {candidate_index}/{len(candidate_paths)}={candidate_path} "
            f"object_frames={len(object_sequence)} action_shape={None if actions is None else actions.shape}; "
            "world-model conditioning uses the recorded hand-surface trajectory derived from those actions"
        )
        predicted_sequence = _rollout_world_model_from_initial_cloud(
            object_sequence=object_sequence,
            hand_points_sequence=hand_points_sequence,
            world_model=wm_setup["model"],
            history_horizon=int(wm_setup["history_horizon"]),
            future_horizon=int(wm_setup["future_horizon"]),
            num_contact_points=int(wm_setup["num_contact_points"]),
            outside_distance_threshold=float(wm_setup["outside_distance_threshold"]),
        )

        ground_truth = object_sequence.numpy()
        hand_points = hand_points_sequence.numpy()
        predicted = predicted_sequence.numpy()
        display_offset = np.asarray(
            [max(float(np.ptp(ground_truth[0], axis=0).max()) * 1.5, 0.15), 0.0, 0.0],
            dtype=np.float32,
        )
        playback_l2_errors = []
        playback_chamfer_errors = []
        for timestep in range(object_sequence.shape[0]):
            if timestep == 0:
                ordered_l2_error = 0.0
                chamfer_error = 0.0
            else:
                pointwise_l2 = torch.linalg.norm(
                    predicted_sequence[timestep, :, :3] - object_sequence[timestep, :, :3], dim=-1
                )
                ordered_l2_error = float(pointwise_l2.mean().item())
                chamfer_error = float(
                    chamfer_distance(
                        predicted_sequence[timestep, :, :3].unsqueeze(0),
                        object_sequence[timestep, :, :3].unsqueeze(0),
                    ).item()
                )
                ordered_l2_errors.append(ordered_l2_error)
                chamfer_errors.append(chamfer_error)
            playback_l2_errors.append(ordered_l2_error)
            playback_chamfer_errors.append(chamfer_error)

            if viewer is not None:
                viewer.update_trajectory_comparison(
                    env_index=0,
                    timestep=timestep,
                    hand_points=hand_points[timestep],
                    ground_truth_points=ground_truth[timestep],
                    predicted_points=predicted[timestep],
                    display_offset=display_offset,
                    ordered_l2_error=ordered_l2_error,
                    chamfer_error=chamfer_error,
                )
            elif timestep == 1 or timestep % max(1, int(wm_cfg.log_interval)) == 0 or timestep == object_sequence.shape[0] - 1:
                print(
                    f"frame={timestep} ordered_l2={ordered_l2_error:.6f} "
                    f"chamfer={chamfer_error:.6f}"
                )

        trajectories.append(
            {
                "candidate_path": candidate_path,
                "ground_truth": ground_truth,
                "hand_points": hand_points,
                "predicted": predicted,
                "display_offset": display_offset,
                "playback_l2_errors": playback_l2_errors,
                "playback_chamfer_errors": playback_chamfer_errors,
                "actions": actions,
                "frame_phase": frame_phase,
            }
        )

    if viewer is not None and bool(wm_cfg.hold_viewer):
        print(
            f"Playback finished for {len(trajectories)} trajectory/trajectories. "
            "Press Enter to replay the batch, or type q and press Enter to close the viewer."
        )
        while True:
            try:
                command = input()
            except EOFError:
                break
            if command.strip().lower() in {"q", "quit", "exit"}:
                break
            for trajectory_index, trajectory in enumerate(trajectories, start=1):
                print(
                    f"replaying candidate {trajectory_index}/{len(trajectories)}: "
                    f"{trajectory['candidate_path']}"
                )
                for timestep in range(len(trajectory["ground_truth"])):
                    viewer.update_trajectory_comparison(
                        env_index=0,
                        timestep=timestep,
                        hand_points=trajectory["hand_points"][timestep],
                        ground_truth_points=trajectory["ground_truth"][timestep],
                        predicted_points=trajectory["predicted"][timestep],
                        display_offset=trajectory["display_offset"],
                        ordered_l2_error=trajectory["playback_l2_errors"][timestep],
                        chamfer_error=trajectory["playback_chamfer_errors"][timestep],
                    )

    overall = {
        "ordered_l2_mean": float(np.mean(ordered_l2_errors)) if ordered_l2_errors else float("nan"),
        "chamfer_mean": float(np.mean(chamfer_errors)) if chamfer_errors else float("nan"),
        "num_eval_steps": len(ordered_l2_errors),
        "num_candidates": len(candidate_paths),
        "candidate_files": [str(path) for path in candidate_paths],
        "candidate_file": str(candidate_paths[0]) if len(candidate_paths) == 1 else None,
        "world_model_checkpoint": str(wm_setup["checkpoint_path"]),
        "recorded_action_shapes": [
            None if trajectory["actions"] is None else tuple(trajectory["actions"].shape)
            for trajectory in trajectories
        ],
        "frame_phase_shapes": [
            None if trajectory["frame_phase"] is None else tuple(trajectory["frame_phase"].shape)
            for trajectory in trajectories
        ],
        "recorded_action_shape": (
            None if trajectories[0]["actions"] is None else tuple(trajectories[0]["actions"].shape)
        ),
        "frame_phase_shape": (
            None if trajectories[0]["frame_phase"] is None else tuple(trajectories[0]["frame_phase"].shape)
        ),
    }
    print("offline world-model metrics:", overall)
    return overall


def _resolve_ppmodel_urdf_path(urdf_path: str) -> str:
    path = Path(urdf_path).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())
    if path.is_absolute():
        parts = path.parts
        if "assets" in parts:
            path = Path(*parts[parts.index("assets"):])
    candidates = (PROJECT_ROOT / path, PROJECT_ROOT.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise FileNotFoundError(f"World-model robot URDF not found: {urdf_path}")


def _build_ppmodel_and_mapping(env, wm_setup: Dict[str, object], preprocess_device: str):
    model = PPModel(
        urdf_path=_resolve_ppmodel_urdf_path(str(wm_setup["urdf_path"])),
        points_per_link=int(wm_setup["points_per_link"]),
        geometry_type="visual",
        root_link_names=wm_setup["root_link_names"],
        device=preprocess_device,
        enable_viewer=False,
    )

    env_joint_name_to_index = {name: idx for idx, name in enumerate(env.robot_dof_names)}
    mapped_model_indices = []
    mapped_env_indices = []
    missing_joint_names = []
    for model_joint_index, joint_name in enumerate(model.get_joint_names()):
        env_joint_index = env_joint_name_to_index.get(joint_name)
        if env_joint_index is None:
            missing_joint_names.append(joint_name)
            continue
        mapped_model_indices.append(model_joint_index)
        mapped_env_indices.append(env_joint_index)

    mapping = {
        "model_joint_indices": torch.tensor(mapped_model_indices, dtype=torch.long, device=model.device),
        "env_joint_indices": torch.tensor(mapped_env_indices, dtype=torch.long, device=env.device),
        "missing_joint_names": missing_joint_names,
    }
    return model, mapping


def _capture_env_state(env, pp_model, model_mapping, env_index: int):
    robot_q = env.robot_dof_pos[env_index].detach().clone()
    model_q = torch.zeros(pp_model.dof, dtype=pp_model.dtype, device=pp_model.device)
    if model_mapping["model_joint_indices"].numel() > 0:
        model_q[model_mapping["model_joint_indices"]] = robot_q[model_mapping["env_joint_indices"]].to(
            device=pp_model.device, dtype=pp_model.dtype
        )

    object_points = env.transform_obj_pcl_2_world()[env_index].detach().clone().cpu()
    return {
        "ppmodel_q": model_q.detach().clone().cpu(),
        "object_points": object_points.to(dtype=torch.float32).contiguous(),
    }


def _record_single_round(
    env,
    pp_model,
    model_mapping,
    env_indices: List[int],
    action_source: str,
    policy=None,
):
    round_episode = {env_index: [] for env_index in env_indices}
    obs = env.reset_idx(torch.arange(env.num_envs))["obs"]
    if policy is not None:
        states = env.get_state()
        with torch.no_grad():
            plan_actions = policy(obs, states, inference=True)
        env.generate_reaching_plan_idx(torch.arange(env.num_envs), actions=plan_actions)

    for env_index in env_indices:
        round_episode[env_index].append(_capture_env_state(env, pp_model, model_mapping, env_index))

    for _ in range(env.max_episode_length):
        if action_source in {"policy", "reference"}:
            env_action = env.compute_reference_actions()
        elif action_source == "zero":
            env_action = env.no_op_action.clone()
        else:
            raise ValueError(f"Unsupported wm.action_source={action_source}")
        env.step(env_action)

        for env_index in env_indices:
            round_episode[env_index].append(_capture_env_state(env, pp_model, model_mapping, env_index))

    round_stats = {
        "success_rate": float(env.successes.float().mean().item()),
        "current_success_rate": float(env.current_successes.float().mean().item()),
    }
    return round_episode, round_stats


def _stack_episode_states(episode_states: List[Dict[str, torch.Tensor]]):
    q_sequence = torch.stack([state["ppmodel_q"] for state in episode_states], dim=0)
    object_sequence = torch.stack([state["object_points"] for state in episode_states], dim=0)
    return q_sequence, object_sequence


def _summarize_per_env_metrics(per_env_episode_metrics: Dict[int, List[Dict[str, float]]]) -> Dict[int, Dict[str, object]]:
    per_env_metrics = {}
    for env_index, episode_metrics in per_env_episode_metrics.items():
        if episode_metrics:
            per_env_metrics[env_index] = {
                "ordered_l2_mean": float(np.mean([metric["ordered_l2_mean"] for metric in episode_metrics])),
                "ordered_l2_max_mean": float(np.mean([metric["ordered_l2_max_mean"] for metric in episode_metrics])),
                "chamfer_mean": float(np.mean([metric["chamfer_mean"] for metric in episode_metrics])),
                "num_eval_steps": int(sum(metric["num_eval_steps"] for metric in episode_metrics)),
                "episodes": list(episode_metrics),
            }
        else:
            per_env_metrics[env_index] = {
                "ordered_l2_mean": float("nan"),
                "ordered_l2_max_mean": float("nan"),
                "chamfer_mean": float("nan"),
                "num_eval_steps": 0,
                "episodes": [],
            }
    return per_env_metrics


def _summarize_overall_metrics(per_env_metrics: Dict[int, Dict[str, object]], checkpoint_path: Path) -> Dict[str, object]:
    if per_env_metrics:
        ordered_l2_values = [float(metrics["ordered_l2_mean"]) for metrics in per_env_metrics.values()]
        ordered_l2_max_values = [float(metrics["ordered_l2_max_mean"]) for metrics in per_env_metrics.values()]
        chamfer_values = [float(metrics["chamfer_mean"]) for metrics in per_env_metrics.values()]
        num_eval_steps = int(sum(int(metrics["num_eval_steps"]) for metrics in per_env_metrics.values()))
    else:
        ordered_l2_values = [float("nan")]
        ordered_l2_max_values = [float("nan")]
        chamfer_values = [float("nan")]
        num_eval_steps = 0

    return {
        "ordered_l2_mean": float(np.mean(ordered_l2_values)),
        "ordered_l2_max_mean": float(np.mean(ordered_l2_max_values)),
        "chamfer_mean": float(np.mean(chamfer_values)),
        "num_eval_steps": num_eval_steps,
        "world_model_checkpoint": str(checkpoint_path),
        "timestamp": datetime.now().isoformat(),
    }


def _evaluate_episode(
    env_index: int,
    *,
    q_sequence: torch.Tensor,
    object_sequence: torch.Tensor,
    pp_model: PPModel,
    world_model: torch.nn.Module,
    history_horizon: int,
    future_horizon: int,
    num_contact_points: int,
    outside_distance_threshold: float,
    viewer: WorldModelViewer | None,
    log_interval: int,
) -> Dict[str, float]:
    predicted_sequence = _rollout_world_model_from_initial_state(
        object_sequence=object_sequence,
        q_sequence=q_sequence,
        pp_model=pp_model,
        world_model=world_model,
        history_horizon=history_horizon,
        future_horizon=future_horizon,
        num_contact_points=num_contact_points,
        outside_distance_threshold=outside_distance_threshold,
    )
    hand_points_sequence = _build_hand_points_sequence(q_sequence=q_sequence, pp_model=pp_model)

    ordered_l2_errors = []
    chamfer_errors = []
    max_l2_errors = []

    for timestep in range(history_horizon, object_sequence.shape[0]):
        predicted_points = predicted_sequence[timestep]
        actual_points = object_sequence[timestep]
        previous_predicted_points = predicted_sequence[timestep - 1]
        display_contact_points = _select_future_contact_points_for_rollout(
            hand_points=hand_points_sequence[timestep : timestep + 1],
            current_object_points=previous_predicted_points,
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
        )[0]

        pointwise_l2 = torch.linalg.norm(predicted_points[..., :3] - actual_points[..., :3], dim=-1)
        ordered_l2_error = float(pointwise_l2.mean().item())
        max_l2_error = float(pointwise_l2.max().item())
        chamfer_error = float(
            chamfer_distance(
                predicted_points[:, :3].unsqueeze(0),
                actual_points[:, :3].unsqueeze(0),
            ).item()
        )

        ordered_l2_errors.append(ordered_l2_error)
        max_l2_errors.append(max_l2_error)
        chamfer_errors.append(chamfer_error)

        if viewer is not None:
            viewer.update(
                env_index=env_index,
                current_object_points=previous_predicted_points.numpy(),
                actual_next_object_points=actual_points.numpy(),
                predicted_next_object_points=predicted_points.numpy(),
                current_contact_points=display_contact_points.numpy(),
                ordered_l2_error=ordered_l2_error,
                chamfer_error=chamfer_error,
            )

        if (
            timestep == history_horizon
            or timestep % max(1, log_interval) == 0
            or timestep == object_sequence.shape[0] - 1
        ):
            print(
                f"env={env_index} open_loop_step={timestep} ordered_l2={ordered_l2_error:.6f} "
                f"max_l2={max_l2_error:.6f} chamfer={chamfer_error:.6f}"
            )

    return {
        "ordered_l2_mean": float(np.mean(ordered_l2_errors)) if ordered_l2_errors else float("nan"),
        "ordered_l2_max_mean": float(np.mean(max_l2_errors)) if max_l2_errors else float("nan"),
        "chamfer_mean": float(np.mean(chamfer_errors)) if chamfer_errors else float("nan"),
        "num_eval_steps": len(ordered_l2_errors),
    }


def _build_contact_sequence(
    *,
    q_sequence: torch.Tensor,
    object_sequence: torch.Tensor,
    pp_model: PPModel,
    num_contact_points: int,
    outside_distance_threshold: float,
) -> torch.Tensor:
    with torch.no_grad():
        q_sequence_device = q_sequence.to(device=pp_model.device, dtype=pp_model.dtype)
        hand_state = pp_model.get_state(q_sequence_device)
        hand_points = hand_state.points
        if hand_points.ndim == 2:
            hand_points = hand_points.unsqueeze(0)
        hand_points = hand_points.detach().cpu().to(dtype=torch.float32)

    contact_sequence = []
    for timestep in range(object_sequence.shape[0]):
        contact_sequence.append(
            _select_contact_points_from_hand_points(
                hand_points[timestep],
                object_sequence[timestep],
                num_contact_points=num_contact_points,
                outside_distance_threshold=outside_distance_threshold,
            )
        )
    return torch.stack(contact_sequence, dim=0)


def _build_hand_points_sequence(*, q_sequence: torch.Tensor, pp_model: PPModel) -> torch.Tensor:
    with torch.no_grad():
        q_sequence_device = q_sequence.to(device=pp_model.device, dtype=pp_model.dtype)
        hand_state = pp_model.get_state(q_sequence_device)
        hand_points = hand_state.points
        if hand_points.ndim == 2:
            hand_points = hand_points.unsqueeze(0)
    return hand_points.detach().cpu().to(dtype=torch.float32)


def _select_future_contact_points_for_rollout(
    *,
    hand_points: torch.Tensor,
    current_object_points: torch.Tensor,
    num_contact_points: int,
    outside_distance_threshold: float,
) -> torch.Tensor:
    future_contact_points = []
    for future_hand_points in hand_points:
        future_contact_points.append(
            _select_contact_points_from_hand_points(
                future_hand_points,
                current_object_points,
                num_contact_points=num_contact_points,
                outside_distance_threshold=outside_distance_threshold,
            )
        )
    return torch.stack(future_contact_points, dim=0)


def _rollout_world_model_from_initial_state(
    *,
    object_sequence: torch.Tensor,
    q_sequence: torch.Tensor,
    pp_model: PPModel,
    world_model: torch.nn.Module,
    history_horizon: int,
    future_horizon: int,
    num_contact_points: int,
    outside_distance_threshold: float,
) -> torch.Tensor:
    if object_sequence.shape[0] <= history_horizon:
        raise ValueError(
            f"Need more than history_horizon={history_horizon} states for open-loop rollout eval, "
            f"got {object_sequence.shape[0]}."
        )

    device = next(world_model.parameters()).device
    start_timestep = history_horizon - 1

    hand_points_sequence = _build_hand_points_sequence(q_sequence=q_sequence, pp_model=pp_model)
    initial_contact_sequence = _build_contact_sequence(
        q_sequence=q_sequence[:history_horizon],
        object_sequence=object_sequence[:history_horizon],
        pp_model=pp_model,
        num_contact_points=num_contact_points,
        outside_distance_threshold=outside_distance_threshold,
    )
    predicted_states = [state.detach().clone() for state in object_sequence[: start_timestep + 1]]
    contact_history_states = [state.detach().clone() for state in initial_contact_sequence]
    next_timestep = start_timestep + 1
    while next_timestep < object_sequence.shape[0]:
        object_history = torch.stack(predicted_states[-history_horizon:], dim=0)
        future_end = min(next_timestep + future_horizon, object_sequence.shape[0])
        future_contact_points = _select_future_contact_points_for_rollout(
            hand_points=hand_points_sequence[next_timestep:future_end],
            current_object_points=predicted_states[-1],
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
        )

        chunk_len = int(future_contact_points.shape[0])
        if chunk_len < future_horizon:
            pad_count = future_horizon - chunk_len
            future_contact_points = torch.cat(
                [future_contact_points, future_contact_points[-1:].repeat(pad_count, 1, 1)],
                dim=0,
            )

        with torch.no_grad():
            prediction = world_model(
                object_history.unsqueeze(0).to(device),
                torch.stack(contact_history_states[-history_horizon:], dim=0).unsqueeze(0).to(device),
                future_contact_points.unsqueeze(0).to(device),
            )

        predicted_chunk = prediction.object_future_points[0, :chunk_len].detach().cpu()
        predicted_states.extend(predicted_chunk[timestep].detach().clone() for timestep in range(chunk_len))
        contact_history_states.extend(future_contact_points[:chunk_len])
        next_timestep += chunk_len

    return torch.stack(predicted_states[: object_sequence.shape[0]], dim=0)


def run_world_model_test(cfg: DictConfig, env) -> Dict[str, object]:
    wm_cfg = _get_wm_cfg(cfg)
    env_indices = _normalize_wm_env_indices(wm_cfg, env.num_envs)
    wm_setup = _resolve_world_model_setup(cfg, wm_cfg)
    num_rounds = _parse_num_rounds(wm_cfg.num_rounds)

    pp_model, model_mapping = _build_ppmodel_and_mapping(
        env,
        wm_setup,
        preprocess_device=str(wm_cfg.preprocess_device),
    )
    if model_mapping["missing_joint_names"]:
        print("Warning: PPModel joints missing in env and set to zero:", model_mapping["missing_joint_names"])

    viewer = None
    if bool(wm_cfg.enable_viewer):
        viewer = WorldModelViewer(
            host=str(wm_cfg.viewer_host),
            port=int(wm_cfg.viewer_port),
            object_point_size=float(wm_cfg.object_point_size),
            contact_point_size=float(wm_cfg.contact_point_size),
            sleep_dt=float(wm_cfg.viewer_sleep_dt),
            show_contact=bool(wm_cfg.show_contact),
        )
        print(f"viewer: http://{wm_cfg.viewer_host}:{int(wm_cfg.viewer_port)}")

    policy = None
    if str(wm_cfg.action_source) == "policy":
        cfg.test = True
        runner = _build_policy_runner(cfg, env)
        policy = runner.actor_critic
        policy.eval()

    if num_rounds is None:
        print("wm.num_rounds is unlimited; press Ctrl+C to stop the rollout loop.")
        round_indices = itertools.count()
    else:
        print(f"wm.num_rounds={num_rounds}")
        round_indices = range(num_rounds)

    rollout_stats = []
    per_env_episode_metrics = {env_index: [] for env_index in env_indices}

    try:
        for round_idx in round_indices:
            episodes, round_stats = _record_single_round(
                env,
                pp_model,
                model_mapping,
                env_indices,
                action_source=str(wm_cfg.action_source),
                policy=policy,
            )

            round_stats["round"] = round_idx
            rollout_stats.append(round_stats)
            print(
                f"round={round_idx} success_rate={round_stats['success_rate']:.4f} "
                f"current_success_rate={round_stats['current_success_rate']:.4f}"
            )

            for env_index in env_indices:
                q_sequence, object_sequence = _stack_episode_states(episodes[env_index])
                metrics = _evaluate_episode(
                    env_index,
                    q_sequence=q_sequence,
                    object_sequence=object_sequence,
                    pp_model=pp_model,
                    world_model=wm_setup["model"],
                    history_horizon=int(wm_setup["history_horizon"]),
                    future_horizon=int(wm_setup["future_horizon"]),
                    num_contact_points=int(wm_setup["num_contact_points"]),
                    outside_distance_threshold=float(wm_setup["outside_distance_threshold"]),
                    viewer=viewer,
                    log_interval=int(wm_cfg.log_interval),
                )
                metrics["round"] = round_idx
                per_env_episode_metrics[env_index].append(metrics)
                print(f"env={env_index} round={round_idx} metrics={metrics}")

            per_env_metrics = _summarize_per_env_metrics(per_env_episode_metrics)
            for env_index in env_indices:
                print(f"env={env_index} aggregated_metrics={per_env_metrics[env_index]}")
    except KeyboardInterrupt:
        print("Interrupted by user, returning metrics collected so far.")

    per_env_metrics = _summarize_per_env_metrics(per_env_episode_metrics)
    overall = _summarize_overall_metrics(per_env_metrics, wm_setup["checkpoint_path"])
    print("overall metrics:", overall)
    return {
        "overall": overall,
        "per_env": per_env_metrics,
        "rollout_stats": rollout_stats,
    }


@hydra.main(version_base="1.3", config_path="../tasks", config_name="config")
def main(cfg: DictConfig) -> None:
    wm_cfg = _get_wm_cfg(cfg)
    if wm_cfg.dataset:
        run_offline_world_model_test(cfg)
        return

    set_np_formatting()

    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    required_num_envs = _infer_required_num_envs_from_wm_cfg(wm_cfg)
    current_num_envs = int(cfg.task.env.numEnvs)
    if required_num_envs is not None and required_num_envs > current_num_envs:
        with open_dict(cfg):
            cfg.task.env.numEnvs = required_num_envs
            if "num_envs" in cfg:
                cfg.num_envs = required_num_envs
        print(
            f"wm.env_indices requires at least {required_num_envs} envs; "
            f"auto-updating num_envs from {current_num_envs} to {required_num_envs}."
        )

    if str(wm_cfg.action_source) == "policy":
        _apply_policy_mode_defaults(cfg)
    else:
        cfg.task.env.enablePointCloud = True

    def create_isaacgym_env(**kwargs):
        return isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

    env = create_isaacgym_env()
    env.reset_idx(torch.arange(env.num_envs))

    if str(wm_cfg.action_source) == "policy":
        _apply_checkpoint_compat_overrides(cfg, env)

    results = run_world_model_test(cfg, env)
    print("trajectory average gap:", results["overall"])


if __name__ == "__main__":
    main()
