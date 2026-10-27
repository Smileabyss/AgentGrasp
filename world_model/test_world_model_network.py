from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from world_model.algo.pn_utils.world_model import (
    build_lightweight_world_model,
    build_trajectory_world_model,
    trajectory_world_model_loss,
    world_model_loss,
)


def _make_synthetic_transition(
    points: torch.Tensor,
    action: torch.Tensor,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    target_next_points = points.clone()
    action_delta = action[:, None, :3].tanh() * 0.05
    target_next_points[..., :3] = target_next_points[..., :3] + action_delta
    target_next_points[..., :3] = target_next_points[..., :3] + noise_scale * torch.randn_like(
        target_next_points[..., :3]
    )
    return target_next_points


def run_single_frame_case(
    *,
    point_dim: int,
    action_dim: int,
    batch_size: int,
    num_points: int,
    rollout_horizon: int,
    device: torch.device,
) -> None:
    print(f"\n=== Single-Frame Test: point_dim={point_dim}, action_dim={action_dim} ===")

    model = build_lightweight_world_model(
        point_dim=point_dim,
        action_dim=action_dim,
        global_feature_dim=128,
        action_feature_dim=128,
        local_feature_dim=128,
        hidden_dim=256,
    ).to(device)
    model.eval()

    points = torch.randn(batch_size, num_points, point_dim, device=device)
    action = torch.randn(batch_size, action_dim, device=device)
    point_mask = torch.rand(batch_size, num_points, device=device) > 0.15
    target_next_points = _make_synthetic_transition(points, action)

    with torch.no_grad():
        prediction = model(points, action, point_mask=point_mask)
        mse_loss = world_model_loss(
            points,
            prediction,
            target_next_points,
            point_mask=point_mask,
            loss_type="ordered_mse",
        )
        chamfer_loss = world_model_loss(
            points,
            prediction,
            target_next_points,
            point_mask=point_mask,
            loss_type="chamfer",
        )
        rollout_actions = torch.randn(batch_size, rollout_horizon, action_dim, device=device)
        rollout_predictions = model.rollout(points, rollout_actions, point_mask=point_mask)

    print("points shape:", tuple(points.shape))
    print("action shape:", tuple(action.shape))
    print("mask valid points:", int(point_mask.sum().item()), "/", point_mask.numel())
    print("delta_points shape:", tuple(prediction.delta_points.shape))
    print("next_points shape:", tuple(prediction.next_points.shape))
    print("global_feature shape:", tuple(prediction.global_feature.shape))
    print("action_feature shape:", tuple(prediction.action_feature.shape))
    print("point_feature shape:", tuple(prediction.point_feature.shape))
    print("ordered_mse loss:", float(mse_loss))
    print("chamfer loss:", float(chamfer_loss))
    print("rollout length:", len(rollout_predictions))
    print("final rollout next_points shape:", tuple(rollout_predictions[-1].next_points.shape))


def _make_synthetic_future_targets(
    object_history: torch.Tensor,
    future_contact_points: torch.Tensor,
    noise_scale: float = 0.005,
) -> torch.Tensor:
    last_object_points = object_history[:, -1]
    contact_center = future_contact_points[..., :3].mean(dim=2, keepdim=True)
    contact_drive = contact_center - future_contact_points[:, :1, :, :3].mean(dim=2, keepdim=True)

    target_object_future = last_object_points[:, None, :, :].repeat(1, future_contact_points.shape[1], 1, 1)
    target_object_future[..., :3] = target_object_future[..., :3] + 0.35 * contact_drive
    target_object_future[..., :3] = target_object_future[..., :3] + noise_scale * torch.randn_like(
        target_object_future[..., :3]
    )
    if target_object_future.shape[-1] > 3:
        target_object_future[..., 3:] = target_object_future[..., 3:] + 0.02
    return target_object_future


def run_trajectory_case(
    *,
    object_point_dim: int,
    contact_point_dim: int,
    batch_size: int,
    history_horizon: int,
    future_horizon: int,
    object_points: int,
    contact_points: int,
    device: torch.device,
) -> None:
    print(
        f"\n=== Trajectory Test: object_point_dim={object_point_dim}, "
        f"contact_point_dim={contact_point_dim} ==="
    )

    model = build_trajectory_world_model(
        object_point_dim=object_point_dim,
        contact_point_dim=contact_point_dim,
        trajectory_feature_dim=128,
        global_feature_dim=128,
        local_feature_dim=128,
        hidden_dim=256,
        recurrent_layers=1,
    ).to(device)
    model.eval()

    object_history = torch.randn(batch_size, history_horizon, object_points, object_point_dim, device=device)
    contact_history = torch.randn(batch_size, history_horizon, contact_points, contact_point_dim, device=device)
    future_contact_points = torch.randn(
        batch_size, future_horizon, contact_points, contact_point_dim, device=device
    )

    object_mask = torch.rand(batch_size, object_points, device=device) > 0.1
    contact_mask = torch.rand(batch_size, contact_points, device=device) > 0.15
    target_object_future = _make_synthetic_future_targets(object_history, future_contact_points)

    with torch.no_grad():
        prediction = model(
            object_history,
            contact_history,
            future_contact_points,
            object_mask=object_mask,
            contact_mask=contact_mask,
        )
        mse_loss = trajectory_world_model_loss(
            prediction,
            target_object_future,
            object_mask=object_mask,
            loss_type="ordered_mse",
        )
        chamfer_loss = trajectory_world_model_loss(
            prediction,
            target_object_future,
            object_mask=object_mask,
            loss_type="chamfer",
        )

    print("object_history shape:", tuple(object_history.shape))
    print("contact_history shape:", tuple(contact_history.shape))
    print("future_contact_points shape:", tuple(future_contact_points.shape))
    print("object_future_points shape:", tuple(prediction.object_future_points.shape))
    print("object_future_delta shape:", tuple(prediction.object_future_delta.shape))
    print("object_history_feature shape:", tuple(prediction.object_history_feature.shape))
    print("contact_history_feature shape:", tuple(prediction.contact_history_feature.shape))
    print("future_contact_feature shape:", tuple(prediction.future_contact_feature.shape))
    print("global_feature shape:", tuple(prediction.global_feature.shape))
    print("ordered_mse loss:", float(mse_loss))
    print("chamfer loss:", float(chamfer_loss))


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Testing lightweight point-flow world model")
    print("project root:", PROJECT_ROOT)
    print("device:", device)

    run_single_frame_case(
        point_dim=3,
        action_dim=6,
        batch_size=2,
        num_points=128,
        rollout_horizon=4,
        device=device,
    )
    run_single_frame_case(
        point_dim=6,
        action_dim=12,
        batch_size=2,
        num_points=256,
        rollout_horizon=3,
        device=device,
    )
    run_trajectory_case(
        object_point_dim=3,
        contact_point_dim=3,
        batch_size=2,
        history_horizon=4,
        future_horizon=5,
        object_points=192,
        contact_points=64,
        device=device,
    )
    run_trajectory_case(
        object_point_dim=6,
        contact_point_dim=6,
        batch_size=2,
        history_horizon=3,
        future_horizon=4,
        object_points=128,
        contact_points=48,
        device=device,
    )

    print("\nAll tests finished.")


if __name__ == "__main__":
    main()
