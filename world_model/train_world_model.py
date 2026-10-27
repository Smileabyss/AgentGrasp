from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_model.algo.pn_utils.world_model import build_trajectory_world_model, trajectory_world_model_loss

try:
    from world_model.PPModel import PPModel
    from world_model.PPModel.contact_surface import _compute_penetration_mask, _downsample_if_needed
except ImportError as exc:
    raise ImportError(
        "train_world_model.py requires PPModel dependencies. "
        "Make sure `pytorch_kinematics` and the PPModel stack are installed in the current environment."
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the trajectory-conditioned point-cloud world model from sampled DemoGrasp rollouts."
    )
    parser.add_argument("--samples-root", type=Path, default=PROJECT_ROOT / "samples")
    parser.add_argument("--latent-search-root", type=Path, default=None)
    parser.add_argument("--sample-glob", type=str, default="**/*.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "samples" / ".world_model_cache",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="assets/inspire_tac/fr3_inspire_tac_L_right_safety.urdf",
    )
    parser.add_argument(
        "--root-link-names",
        type=str,
        default="base_link",
        help="Comma-separated PPModel root link names. Use 'None' for the full robot.",
    )
    parser.add_argument("--points-per-link", type=int, default=64)
    parser.add_argument("--num-contact-points", type=int, default=64)
    parser.add_argument("--outside-distance-threshold", type=float, default=0.01)
    parser.add_argument("--history-horizon", type=int, default=4)
    parser.add_argument("--future-horizon", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preprocess-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--loss-type", type=str, default="ordered_mse", choices=["ordered_mse", "chamfer"])
    parser.add_argument("--trajectory-feature-dim", type=int, default=128)
    parser.add_argument("--global-feature-dim", type=int, default=128)
    parser.add_argument("--local-feature-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=20)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_root_link_names(raw_value: str):
    if raw_value.strip().lower() in {"", "none", "null"}:
        return None
    names = [name.strip() for name in raw_value.split(",") if name.strip()]
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return names


def _resolve_sample_files(samples_root: Path, sample_glob: str, max_files: int) -> List[Path]:
    sample_files = sorted(path for path in samples_root.glob(sample_glob) if path.is_file())
    if max_files > 0:
        sample_files = sample_files[:max_files]
    if not sample_files:
        raise FileNotFoundError(f"No sample files found under {samples_root} with pattern {sample_glob!r}")
    return sample_files


def _cache_file_path(
    sample_path: Path,
    cache_dir: Path,
    *,
    urdf_path: str,
    root_link_names,
    points_per_link: int,
    num_contact_points: int,
    outside_distance_threshold: float,
    preprocessing_version: str = "v1",
) -> Path:
    key = {
        "sample_path": str(sample_path.resolve()),
        "mtime_ns": sample_path.stat().st_mtime_ns,
        "urdf_path": urdf_path,
        "root_link_names": root_link_names,
        "points_per_link": int(points_per_link),
        "num_contact_points": int(num_contact_points),
        "outside_distance_threshold": float(outside_distance_threshold),
        "preprocessing_version": preprocessing_version,
    }
    digest = hashlib.md5(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / f"{sample_path.stem}_{digest}.pt"


def _map_sample_q_to_model_q(
    robot_dof_pos: np.ndarray,
    sample_joint_names: Sequence[str],
    model_joint_names: Sequence[str],
) -> np.ndarray:
    sample_index_by_name = {str(name): index for index, name in enumerate(sample_joint_names)}
    missing_names = [joint_name for joint_name in model_joint_names if joint_name not in sample_index_by_name]
    if missing_names:
        raise ValueError(f"Sample is missing joints required by PPModel: {missing_names}")
    model_indices = [sample_index_by_name[joint_name] for joint_name in model_joint_names]
    return robot_dof_pos[:, model_indices]


def _repeat_to_fixed_count(points: torch.Tensor, target_count: int) -> torch.Tensor:
    if points.shape[0] == target_count:
        return points
    if points.shape[0] == 0:
        raise ValueError("Cannot pad an empty point cloud to a fixed count.")
    if points.shape[0] > target_count:
        return _downsample_if_needed(points, target_count)

    repeat_factor = math.ceil(target_count / points.shape[0])
    repeated = points.repeat(repeat_factor, 1)
    return repeated[:target_count]


def _select_contact_points_from_hand_points(
    hand_points: torch.Tensor,
    object_points: torch.Tensor,
    num_contact_points: int,
    outside_distance_threshold: float,
) -> torch.Tensor:
    hand_xyz = hand_points[:, :3]
    object_xyz = object_points[:, :3]
    min_distances = torch.cdist(hand_xyz, object_xyz, p=2).min(dim=-1).values

    penetration_mask = _compute_penetration_mask(hand_xyz, object_xyz)
    if penetration_mask.any():
        outside_near_mask = (~penetration_mask) & (min_distances < outside_distance_threshold)
        candidate_points = hand_points[penetration_mask | outside_near_mask]
        if candidate_points.shape[0] == 0:
            nearest_indices = torch.argsort(min_distances)[:num_contact_points]
            candidate_points = hand_points[nearest_indices]
    else:
        nearest_indices = torch.argsort(min_distances)[:num_contact_points]
        candidate_points = hand_points[nearest_indices]

    candidate_points = _repeat_to_fixed_count(candidate_points, num_contact_points)
    return candidate_points[:, :3].to(dtype=torch.float32)


def _build_episode_cache(
    sample_path: Path,
    cache_path: Path,
    *,
    pp_model: PPModel,
    num_contact_points: int,
    outside_distance_threshold: float,
    success_only: bool,
) -> Dict[str, object] | None:
    sample = np.load(sample_path, allow_pickle=True)
    success = bool(np.asarray(sample["success"]).item())
    if success_only and not success:
        return None

    robot_dof_pos = np.asarray(sample["robot_dof_pos"], dtype=np.float32)
    object_point_cloud = np.asarray(sample["object_point_cloud_world"], dtype=np.float32)
    sample_joint_names = [str(name) for name in np.asarray(sample["robot_dof_names"]).tolist()]
    model_q = _map_sample_q_to_model_q(robot_dof_pos, sample_joint_names, pp_model.get_joint_names())

    q_tensor = torch.as_tensor(model_q, dtype=pp_model.dtype, device=pp_model.device)
    hand_state = pp_model.get_state(q_tensor)
    hand_points = hand_state.points
    if hand_points.ndim == 2:
        hand_points = hand_points.unsqueeze(0)
    hand_points = hand_points.detach().to(dtype=torch.float32).cpu()
    object_points = torch.as_tensor(object_point_cloud, dtype=torch.float32)

    contact_history = []
    for timestep in range(object_points.shape[0]):
        contact_points = _select_contact_points_from_hand_points(
            hand_points[timestep],
            object_points[timestep],
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
        )
        contact_history.append(contact_points)

    cache_payload = {
        "source_path": str(sample_path.resolve()),
        "success": success,
        "object_points": object_points.contiguous(),
        "hand_points": hand_points.contiguous(),
        "contact_points": torch.stack(contact_history, dim=0).contiguous(),
        "object_asset_file": str(np.asarray(sample["object_asset_file"]).item()),
        "robot_joint_names": list(pp_model.get_joint_names()),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, cache_path)
    return cache_payload


def _prepare_episode_caches(
    sample_files: Sequence[Path],
    *,
    pp_model: PPModel,
    cache_dir: Path,
    urdf_path: str,
    root_link_names,
    points_per_link: int,
    num_contact_points: int,
    outside_distance_threshold: float,
    success_only: bool,
    rebuild_cache: bool,
) -> List[Path]:
    cache_paths = []
    for index, sample_path in enumerate(sample_files, start=1):
        cache_path = _cache_file_path(
            sample_path,
            cache_dir,
            urdf_path=urdf_path,
            root_link_names=root_link_names,
            points_per_link=points_per_link,
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
        )
        try:
            if rebuild_cache or not cache_path.exists():
                print(f"[cache] {index}/{len(sample_files)} building {sample_path}")
                payload = _build_episode_cache(
                    sample_path,
                    cache_path,
                    pp_model=pp_model,
                    num_contact_points=num_contact_points,
                    outside_distance_threshold=outside_distance_threshold,
                    success_only=success_only,
                )
                if payload is None:
                    continue
            else:
                payload = torch.load(cache_path, map_location="cpu")
                if success_only and not bool(payload["success"]):
                    continue
            cache_paths.append(cache_path)
        except Exception as exc:
            print(f"[cache] skipping {sample_path} because {exc}")

    if not cache_paths:
        raise RuntimeError("No cache files were produced. Check the sample directory or filtering options.")
    return cache_paths


def _build_latent_search_episode_cache(
    sample_path: Path,
    cache_path: Path,
    *,
    num_contact_points: int,
    outside_distance_threshold: float,
    success_only: bool,
) -> Dict[str, object] | None:
    with np.load(sample_path, allow_pickle=False) as sample:
        required = ("object_points_base", "hand_points_base", "robot_base_link")
        missing = [key for key in required if key not in sample]
        if missing:
            raise KeyError(f"{sample_path} is missing required fields: {missing}")

        object_points = np.asarray(sample["object_points_base"], dtype=np.float32)
        hand_points = np.asarray(sample["hand_points_base"], dtype=np.float32)
        robot_base_link = str(np.asarray(sample["robot_base_link"]).item())
        success = bool(np.asarray(sample["valid_success"]).item()) if "valid_success" in sample else False
        robot_joint_names = [str(name) for name in np.asarray(sample["robot_dof_names"]).tolist()]
        object_asset_file = (
            str(np.asarray(sample["object_asset_file"]).item()) if "object_asset_file" in sample else ""
        )

    if object_points.ndim != 3 or object_points.shape[-1] != 3:
        raise ValueError(f"object_points_base must have shape (T, N, 3), got {object_points.shape}")
    frame_count = object_points.shape[0]
    if hand_points.ndim != 3 or hand_points.shape[0] != frame_count or hand_points.shape[-1] != 3:
        raise ValueError("hand_points_base must have shape (T, N, 3) and match object_points_base")
    if not robot_base_link:
        raise ValueError("robot_base_link must identify the frame used for the point clouds")
    if success_only and not success:
        return None

    object_points_tensor = torch.as_tensor(object_points, dtype=torch.float32)
    hand_points_tensor = torch.as_tensor(hand_points, dtype=torch.float32)
    contact_history = []
    for timestep in range(frame_count):
        contact_history.append(
            _select_contact_points_from_hand_points(
                hand_points_tensor[timestep],
                object_points_tensor[timestep],
                num_contact_points=num_contact_points,
                outside_distance_threshold=outside_distance_threshold,
            )
        )

    cache_payload = {
        "source_path": str(sample_path.resolve()),
        "success": success,
        "object_points": object_points_tensor.contiguous(),
        "hand_points": hand_points_tensor.contiguous(),
        "contact_points": torch.stack(contact_history, dim=0).contiguous(),
        "robot_base_link": robot_base_link,
        "object_asset_file": object_asset_file,
        "robot_joint_names": robot_joint_names,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, cache_path)
    return cache_payload


def _prepare_latent_search_episode_caches(
    sample_files: Sequence[Path],
    *,
    cache_dir: Path,
    num_contact_points: int,
    outside_distance_threshold: float,
    success_only: bool,
    rebuild_cache: bool,
) -> List[Path]:
    cache_paths = []
    for index, sample_path in enumerate(sample_files, start=1):
        cache_path = _cache_file_path(
            sample_path,
            cache_dir,
            urdf_path="latent_search_rgbd",
            root_link_names=None,
            points_per_link=0,
            num_contact_points=num_contact_points,
            outside_distance_threshold=outside_distance_threshold,
            preprocessing_version="robot_base_v2",
        )
        try:
            if rebuild_cache or not cache_path.exists():
                print(f"[cache] {index}/{len(sample_files)} building {sample_path}")
                payload = _build_latent_search_episode_cache(
                    sample_path,
                    cache_path,
                    num_contact_points=num_contact_points,
                    outside_distance_threshold=outside_distance_threshold,
                    success_only=success_only,
                )
                if payload is None:
                    continue
            else:
                payload = torch.load(cache_path, map_location="cpu")
                if success_only and not bool(payload["success"]):
                    continue
            cache_paths.append(cache_path)
        except Exception as exc:
            print(f"[cache] skipping {sample_path} because {exc}")

    if not cache_paths:
        raise RuntimeError("No cache files were produced. Check the latent-search directory or filtering options.")
    return cache_paths


class TrajectoryWindowDataset(Dataset):
    def __init__(
        self,
        episode_cache_paths: Sequence[Path],
        *,
        history_horizon: int,
        future_horizon: int,
        num_contact_points: int,
        outside_distance_threshold: float,
        lru_cache_size: int = 8,
    ):
        self.episode_cache_paths = [Path(path) for path in episode_cache_paths]
        self.history_horizon = int(history_horizon)
        self.future_horizon = int(future_horizon)
        self.num_contact_points = int(num_contact_points)
        self.outside_distance_threshold = float(outside_distance_threshold)
        self.lru_cache_size = int(max(1, lru_cache_size))
        self._episode_cache: OrderedDict[str, Dict[str, object]] = OrderedDict()
        self._window_index = []

        for episode_idx, cache_path in enumerate(self.episode_cache_paths):
            payload = torch.load(cache_path, map_location="cpu")
            num_states = int(payload["object_points"].shape[0])
            max_current_timestep = num_states - self.future_horizon - 1
            for current_timestep in range(self.history_horizon - 1, max_current_timestep + 1):
                self._window_index.append((episode_idx, current_timestep))

        if not self._window_index:
            raise RuntimeError(
                "No valid trajectory windows were found. "
                "Try reducing history/future horizon or providing longer trajectories."
            )

    def __len__(self) -> int:
        return len(self._window_index)

    def _load_episode(self, episode_idx: int) -> Dict[str, object]:
        cache_path = self.episode_cache_paths[episode_idx]
        cache_key = str(cache_path.resolve())
        if cache_key in self._episode_cache:
            payload = self._episode_cache.pop(cache_key)
            self._episode_cache[cache_key] = payload
            return payload

        payload = torch.load(cache_path, map_location="cpu")
        self._episode_cache[cache_key] = payload
        while len(self._episode_cache) > self.lru_cache_size:
            self._episode_cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int):
        episode_idx, current_timestep = self._window_index[index]
        payload = self._load_episode(episode_idx)

        object_points = payload["object_points"]
        hand_points = payload["hand_points"]
        contact_points = payload["contact_points"]

        history_start = current_timestep - self.history_horizon + 1
        history_end = current_timestep + 1
        future_end = current_timestep + 1 + self.future_horizon

        object_history = object_points[history_start:history_end]
        contact_history = contact_points[history_start:history_end]
        current_object_points = object_points[current_timestep]
        future_hand_points = hand_points[current_timestep + 1:future_end]

        future_contact_points = []
        for future_hand in future_hand_points:
            future_contact_points.append(
                _select_contact_points_from_hand_points(
                    future_hand,
                    current_object_points,
                    num_contact_points=self.num_contact_points,
                    outside_distance_threshold=self.outside_distance_threshold,
                )
            )

        target_object_future = object_points[current_timestep + 1:future_end]

        return {
            "object_history": object_history.to(dtype=torch.float32),
            "contact_history": contact_history.to(dtype=torch.float32),
            "future_contact_points": torch.stack(future_contact_points, dim=0).to(dtype=torch.float32),
            "target_object_future": target_object_future.to(dtype=torch.float32),
        }


def _split_episode_paths(cache_paths: Sequence[Path], train_split: float, seed: int):
    cache_paths = list(cache_paths)
    rng = random.Random(seed)
    rng.shuffle(cache_paths)

    if len(cache_paths) == 1:
        return cache_paths, cache_paths

    split_index = max(1, min(len(cache_paths) - 1, int(round(len(cache_paths) * train_split))))
    return cache_paths[:split_index], cache_paths[split_index:]


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    optimizer,
    device: torch.device,
    loss_type: str,
    train: bool,
    grad_clip: float,
    print_freq: int,
) -> float:
    model.train(train)
    losses = []

    for step, batch in enumerate(dataloader, start=1):
        object_history = batch["object_history"].to(device)
        contact_history = batch["contact_history"].to(device)
        future_contact_points = batch["future_contact_points"].to(device)
        target_object_future = batch["target_object_future"].to(device)

        with torch.set_grad_enabled(train):
            prediction = model(
                object_history,
                contact_history,
                future_contact_points,
            )
            loss = trajectory_world_model_loss(
                prediction,
                target_object_future,
                object_mask=None,
                loss_type=loss_type,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        if step % max(1, print_freq) == 0:
            split_name = "train" if train else "val"
            print(f"[{split_name}] step={step}/{len(dataloader)} loss={loss_value:.6f}")

    return _mean(losses)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    def _to_jsonable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): _to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(item) for item in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=True))


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "runs_world_model" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "config.json", vars(args))

    root_link_names = _parse_root_link_names(args.root_link_names)
    latent_search_mode = args.latent_search_root is not None
    samples_root = args.latent_search_root if latent_search_mode else args.samples_root
    sample_files = _resolve_sample_files(samples_root, args.sample_glob, args.max_files)
    print("sample files:", len(sample_files))

    if latent_search_mode:
        cache_paths = _prepare_latent_search_episode_caches(
            sample_files,
            cache_dir=args.cache_dir,
            num_contact_points=args.num_contact_points,
            outside_distance_threshold=args.outside_distance_threshold,
            success_only=args.success_only,
            rebuild_cache=args.rebuild_cache,
        )
    else:
        pp_model = PPModel(
            urdf_path=args.urdf_path,
            points_per_link=args.points_per_link,
            geometry_type="visual",
            device=args.preprocess_device,
            root_link_names=root_link_names,
            enable_viewer=False,
        )
        print("ppmodel joints:", pp_model.get_joint_names())
        print("ppmodel links:", pp_model.get_link_names())
        cache_paths = _prepare_episode_caches(
            sample_files,
            pp_model=pp_model,
            cache_dir=args.cache_dir,
            urdf_path=args.urdf_path,
            root_link_names=root_link_names,
            points_per_link=args.points_per_link,
            num_contact_points=args.num_contact_points,
            outside_distance_threshold=args.outside_distance_threshold,
            success_only=args.success_only,
            rebuild_cache=args.rebuild_cache,
        )
    print("usable cached episodes:", len(cache_paths))

    if args.cache_only:
        print("cache-only mode finished.")
        return

    train_cache_paths, val_cache_paths = _split_episode_paths(cache_paths, args.train_split, args.seed)
    train_dataset = TrajectoryWindowDataset(
        train_cache_paths,
        history_horizon=args.history_horizon,
        future_horizon=args.future_horizon,
        num_contact_points=args.num_contact_points,
        outside_distance_threshold=args.outside_distance_threshold,
    )
    val_dataset = TrajectoryWindowDataset(
        val_cache_paths,
        history_horizon=args.history_horizon,
        future_horizon=args.future_horizon,
        num_contact_points=args.num_contact_points,
        outside_distance_threshold=args.outside_distance_threshold,
    )

    print("train episodes:", len(train_cache_paths), "train windows:", len(train_dataset))
    print("val episodes:", len(val_cache_paths), "val windows:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = build_trajectory_world_model(
        object_point_dim=3,
        contact_point_dim=3,
        trajectory_feature_dim=args.trajectory_feature_dim,
        global_feature_dim=args.global_feature_dim,
        local_feature_dim=args.local_feature_dim,
        hidden_dim=args.hidden_dim,
        recurrent_layers=args.recurrent_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        if latent_search_mode and checkpoint.get("point_cloud_frame") != "robot_base":
            raise ValueError(
                "Cannot resume a checkpoint without robot-base-frame training metadata. "
                "Train a new model from scratch with the robot-base dataset."
            )
        checkpoint_args = checkpoint.get("args", {})
        architecture_args = (
            "trajectory_feature_dim",
            "global_feature_dim",
            "local_feature_dim",
            "hidden_dim",
            "recurrent_layers",
        )
        mismatches = [
            f"{name}: checkpoint={checkpoint_args[name]}, current={getattr(args, name)}"
            for name in architecture_args
            if name in checkpoint_args and int(checkpoint_args[name]) != int(getattr(args, name))
        ]
        if mismatches:
            raise ValueError("Resume checkpoint architecture does not match: " + "; ".join(mismatches))
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"resumed checkpoint: {resume_path} (epoch {start_epoch})")

    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
            train_loss = _run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                loss_type=args.loss_type,
                train=True,
                grad_clip=args.grad_clip,
                print_freq=args.print_freq,
            )
            val_loss = _run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_type=args.loss_type,
                train=False,
                grad_clip=args.grad_clip,
                print_freq=args.print_freq,
            )

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            metrics_file.write(json.dumps(metrics) + "\n")
            metrics_file.flush()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "root_link_names": root_link_names,
                "point_cloud_frame": "robot_base" if latent_search_mode else None,
            }
            torch.save(checkpoint, output_dir / "latest.pt")
            if is_best:
                torch.save(checkpoint, output_dir / "best.pt")

            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f}"
            )

    print("training finished")
    print("best checkpoint:", output_dir / "best.pt")


if __name__ == "__main__":
    main()
