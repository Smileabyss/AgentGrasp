"""Checkpoint loading helpers used by world-model inference."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_checkpoint_path(path_like, *, label: str = "checkpoint") -> Path:
    raw_path = Path(str(path_like)).expanduser()
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.extend((PROJECT_ROOT / raw_path, PROJECT_ROOT.parent / raw_path))
    else:
        parts = raw_path.parts
        for marker in ("assets", "ckpt", "runs_world_model"):
            if marker in parts:
                suffix = Path(*parts[parts.index(marker):])
                candidates.extend((PROJECT_ROOT / suffix, PROJECT_ROOT.parent / suffix))
                break
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{label.capitalize()} not found: {raw_path}")


def _apply_checkpoint_compat_overrides(cfg: DictConfig, env):
    ckpt_path = cfg.checkpoint
    if not ckpt_path or str(cfg.train.params.name).lower() != "ppo_onestep":
        return

    import torch

    ckpt_path = str(_resolve_checkpoint_path(ckpt_path, label="policy checkpoint"))
    cfg.checkpoint = ckpt_path
    state_dict = torch.load(ckpt_path, map_location="cpu")
    uses_vision = any(key.startswith("backbone.") for key in state_dict.keys())
    ckpt_action_dim = int(state_dict["log_std"].numel()) if "log_std" in state_dict else None
    updates = []

    if uses_vision and not bool(cfg.train.params.is_vision):
        cfg.train.params.is_vision = True
        updates.append("train.params.is_vision=True")
    if not env.randomize_tracking_reference:
        env.randomize_tracking_reference = True
        cfg.task.env.randomizeTrackingReference = True
        updates.append("task.env.randomizeTrackingReference=True")

    if ckpt_action_dim is not None:
        with_grasp_dim = 6 + int(env.num_active_hand_dofs)
        if ckpt_action_dim == with_grasp_dim and not env.randomize_grasp_pose:
            env.randomize_grasp_pose = True
            cfg.task.env.randomizeGraspPose = True
            updates.append("task.env.randomizeGraspPose=True")
        elif ckpt_action_dim == 6 and env.randomize_grasp_pose:
            env.randomize_grasp_pose = False
            cfg.task.env.randomizeGraspPose = False
            updates.append("task.env.randomizeGraspPose=False")
        elif ckpt_action_dim not in {6, with_grasp_dim}:
            raise ValueError(
                f"Checkpoint action dim {ckpt_action_dim} is incompatible with the current hand "
                f"(expected 6 or {with_grasp_dim})."
            )
    if updates:
        print("Auto-applied checkpoint compatibility overrides:", ", ".join(updates))


def _build_policy_runner(cfg: DictConfig, env):
    from memory.policy_inference import load_checkpoint_policy

    return load_checkpoint_policy(cfg, env)
