"""Load the legacy one-step grasp policy only for dataset generation."""

from __future__ import annotations

from types import SimpleNamespace


def load_checkpoint_policy(cfg, env):
    """Build the saved policy architecture and load a checkpoint for inference."""
    import torch
    from algo.ppo_onestep.module import ActorCritic

    if not cfg.checkpoint:
        raise ValueError("A trained checkpoint is required for grasp data collection")
    train_params = cfg.train.params
    if str(train_params.name).lower() != "ppo_onestep":
        raise ValueError(f"Unsupported checkpoint policy: {train_params.name}")

    action_dim = 6 + (env.num_active_hand_dofs if env.randomize_grasp_pose else 0)
    actor_critic = ActorCritic(
        env.observation_space.shape,
        env.state_space.shape,
        (action_dim,),
        train_params.get("init_noise_std", 0.3),
        train_params.policy,
        asymmetric=env.num_states > 0,
        use_pcl=bool(train_params.is_vision),
    ).to(env.device)
    actor_critic.load_state_dict(torch.load(cfg.checkpoint, map_location=env.device))
    actor_critic.eval()
    return SimpleNamespace(actor_critic=actor_critic)
