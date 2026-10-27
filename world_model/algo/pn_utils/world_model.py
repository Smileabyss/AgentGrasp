from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

try:
    from algo.pn_utils.maniskill_learn.networks.backbones.mlp import ConvMLP as ManiSkillConvMLP
    from algo.pn_utils.maniskill_learn.networks.backbones.mlp import LinearMLP as ManiSkillLinearMLP
    from algo.pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet as get_maniskill_pointnet
except Exception:
    ManiSkillConvMLP = None
    ManiSkillLinearMLP = None
    get_maniskill_pointnet = None

from .rnn import RNNLayer




@dataclass
class PointCloudTransitionPrediction:
    delta_points: torch.Tensor
    next_points: torch.Tensor
    global_feature: torch.Tensor
    action_feature: torch.Tensor
    point_feature: torch.Tensor
    log_variance: Optional[torch.Tensor] = None


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean()

    if mask.ndim == tensor.ndim - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=tensor.dtype, device=tensor.device).expand_as(tensor)
    denom = mask.sum().clamp_min(1.0)
    return (tensor * mask).sum() / denom


class _FallbackLinearMLP(nn.Module):
    def __init__(self, mlp_spec: Sequence[int], inactivated_output: bool = True):
        super().__init__()
        layers = []
        for index in range(len(mlp_spec) - 1):
            in_dim, out_dim = int(mlp_spec[index]), int(mlp_spec[index + 1])
            layers.append(nn.Linear(in_dim, out_dim))
            is_output = index == len(mlp_spec) - 2
            if not (is_output and inactivated_output):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _FallbackConvMLP(nn.Module):
    def __init__(self, mlp_spec: Sequence[int], inactivated_output: bool = True):
        super().__init__()
        layers = []
        for index in range(len(mlp_spec) - 1):
            in_dim, out_dim = int(mlp_spec[index]), int(mlp_spec[index + 1])
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            is_output = index == len(mlp_spec) - 2
            if not (is_output and inactivated_output):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _FallbackPointNetEncoder(nn.Module):
    def __init__(self, input_feature_dim: int, feat_dim: int):
        super().__init__()
        self.input_feature_dim = int(input_feature_dim)
        self.feat_dim = int(feat_dim)
        self.local_net = _FallbackConvMLP(
            [self.input_feature_dim + 3, 128, 256],
            inactivated_output=False,
        )
        self.global_net = _FallbackLinearMLP([256, 256, self.feat_dim], inactivated_output=False)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        coords = points[..., :3]
        centered_coords = coords - coords.mean(dim=1, keepdim=True)
        augmented_points = torch.cat([centered_coords, points], dim=-1)
        point_feature = self.local_net(augmented_points.transpose(1, 2))
        pooled_feature = point_feature.max(dim=-1).values
        return self.global_net(pooled_feature)


def _build_linear_mlp(mlp_spec: Sequence[int], inactivated_output: bool) -> nn.Module:
    if ManiSkillLinearMLP is not None:
        return ManiSkillLinearMLP(
            mlp_spec=list(mlp_spec),
            norm_cfg=None,
            bias="auto",
            inactivated_output=inactivated_output,
        )
    return _FallbackLinearMLP(mlp_spec=list(mlp_spec), inactivated_output=inactivated_output)


def _build_conv_mlp(mlp_spec: Sequence[int], inactivated_output: bool) -> nn.Module:
    if ManiSkillConvMLP is not None:
        return ManiSkillConvMLP(
            mlp_spec=list(mlp_spec),
            norm_cfg=None,
            bias="auto",
            inactivated_output=inactivated_output,
        )
    return _FallbackConvMLP(mlp_spec=list(mlp_spec), inactivated_output=inactivated_output)


def _build_global_encoder(point_dim: int, feat_dim: int) -> nn.Module:
    if get_maniskill_pointnet is not None:
        return get_maniskill_pointnet(
            {
                "input_feature_dim": int(point_dim),
                "feat_dim": int(feat_dim),
            }
        )
    return _FallbackPointNetEncoder(input_feature_dim=point_dim, feat_dim=feat_dim)


def chamfer_distance(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    source_mask: Optional[torch.Tensor] = None,
    target_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if source_points.ndim != 3 or target_points.ndim != 3:
        raise ValueError("source_points and target_points must have shape (B, N, 3) and (B, M, 3)")

    dist = torch.cdist(source_points, target_points, p=2).pow(2)

    if source_mask is not None:
        source_mask = source_mask.to(device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(~source_mask[:, :, None], 0.0)
    if target_mask is not None:
        target_mask = target_mask.to(device=dist.device, dtype=torch.bool)
        dist = dist.masked_fill(~target_mask[:, None, :], float("inf"))

    src_to_tgt = dist.min(dim=2).values
    tgt_to_src = dist.min(dim=1).values
    if source_mask is not None:
        src_to_tgt = src_to_tgt.masked_fill(~source_mask, 0.0)
    if target_mask is not None:
        tgt_to_src = tgt_to_src.masked_fill(~target_mask, 0.0)

    src_loss = masked_mean(src_to_tgt.unsqueeze(-1), source_mask)
    tgt_loss = masked_mean(tgt_to_src.unsqueeze(-1), target_mask)
    return src_loss + tgt_loss


def world_model_loss(
    current_points: torch.Tensor,
    prediction: PointCloudTransitionPrediction,
    target_next_points: torch.Tensor,
    point_mask: Optional[torch.Tensor] = None,
    loss_type: str = "ordered_mse",
    delta_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
) -> torch.Tensor:
    if current_points.shape != prediction.next_points.shape:
        raise ValueError(
            f"current_points shape {tuple(current_points.shape)} must match "
            f"prediction.next_points shape {tuple(prediction.next_points.shape)}"
        )
    if target_next_points.shape != prediction.next_points.shape:
        raise ValueError(
            f"target_next_points shape {tuple(target_next_points.shape)} must match "
            f"prediction.next_points shape {tuple(prediction.next_points.shape)}"
        )

    if loss_type == "ordered_mse":
        target_delta = target_next_points[..., :3] - current_points[..., :3]
        delta_error = (prediction.delta_points - target_delta).pow(2)
        recon_error = (prediction.next_points[..., :3] - target_next_points[..., :3]).pow(2)
        return (
            delta_weight * masked_mean(delta_error, point_mask)
            + reconstruction_weight * masked_mean(recon_error, point_mask)
        )

    if loss_type == "chamfer":
        return chamfer_distance(
            prediction.next_points[..., :3],
            target_next_points[..., :3],
            source_mask=point_mask,
            target_mask=point_mask,
        )

    raise ValueError(f"Unsupported loss_type: {loss_type}")


class LightweightPointFlowWorldModel(nn.Module):
    """
    A lightweight action-conditioned point-flow model.

    Design notes:
    - Reuses the local PointNet backbone already shipped in this repo for a compact scene summary.
    - Predicts per-point `delta_xyz` like PointWorld's point-flow target, but keeps the decoder small.
    - Stays much simpler than DreamDojo-style latent video world models so it can be trained inside this codebase.
    - Falls back to pure PyTorch PointNet/MLP blocks when the heavier ManiSkill-style stack is unavailable.
    """

    def __init__(
        self,
        point_dim: int = 3,
        action_dim: int = 0,
        global_feature_dim: int = 128,
        action_feature_dim: int = 128,
        local_feature_dim: int = 128,
        hidden_dim: int = 256,
        predict_log_variance: bool = False,
        action_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        if point_dim < 3:
            raise ValueError("point_dim must be at least 3 because xyz coordinates are required.")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive.")

        self.point_dim = int(point_dim)
        self.action_dim = int(action_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.action_feature_dim = int(action_feature_dim)
        self.local_feature_dim = int(local_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.predict_log_variance = bool(predict_log_variance)

        self.global_encoder = _build_global_encoder(self.point_dim, self.global_feature_dim)
        self.local_encoder = _build_conv_mlp(
            mlp_spec=[self.point_dim, self.local_feature_dim, self.local_feature_dim],
            inactivated_output=False,
        )

        action_layers = list(action_layers) if action_layers is not None else [self.action_feature_dim]
        self.action_encoder = _build_linear_mlp(
            mlp_spec=[self.action_dim] + action_layers + [self.action_feature_dim],
            inactivated_output=False,
        )

        fusion_dim = self.point_dim + self.local_feature_dim + self.global_feature_dim + self.action_feature_dim
        self.fusion_encoder = _build_conv_mlp(
            mlp_spec=[fusion_dim, self.hidden_dim, self.hidden_dim],
            inactivated_output=False,
        )
        self.flow_head = _build_conv_mlp(
            mlp_spec=[self.hidden_dim, self.hidden_dim, 3],
            inactivated_output=True,
        )

        if self.predict_log_variance:
            self.log_variance_head = _build_conv_mlp(
                mlp_spec=[self.hidden_dim, self.hidden_dim, 3],
                inactivated_output=True,
            )
        else:
            self.log_variance_head = None

    def forward(
        self,
        points: torch.Tensor,
        action: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> PointCloudTransitionPrediction:
        if points.ndim != 3:
            raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
        if action.ndim != 2:
            raise ValueError(f"action must have shape (B, A), got {tuple(action.shape)}")
        if points.shape[-1] != self.point_dim:
            raise ValueError(f"Expected point_dim={self.point_dim}, got points.shape[-1]={points.shape[-1]}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got action.shape[-1]={action.shape[-1]}")
        if points.shape[0] != action.shape[0]:
            raise ValueError("points and action batch size must match.")

        batch_size, num_points, _ = points.shape
        if point_mask is None:
            point_mask = torch.ones((batch_size, num_points), dtype=torch.bool, device=points.device)
        else:
            if point_mask.shape != (batch_size, num_points):
                raise ValueError(
                    f"point_mask must have shape {(batch_size, num_points)}, got {tuple(point_mask.shape)}"
                )
            point_mask = point_mask.to(device=points.device, dtype=torch.bool)

        global_feature = self.global_encoder(points)
        local_feature = self.local_encoder(points.transpose(1, 2)).transpose(1, 2)
        action_feature = self.action_encoder(action)

        repeated_global = global_feature.unsqueeze(1).expand(-1, num_points, -1)
        repeated_action = action_feature.unsqueeze(1).expand(-1, num_points, -1)
        fused_feature = torch.cat([points, local_feature, repeated_global, repeated_action], dim=-1)
        fused_feature = self.fusion_encoder(fused_feature.transpose(1, 2)).transpose(1, 2)

        delta_points = self.flow_head(fused_feature.transpose(1, 2)).transpose(1, 2)
        delta_points = delta_points * point_mask.unsqueeze(-1).to(delta_points.dtype)

        next_points = points.clone()
        next_points[..., :3] = next_points[..., :3] + delta_points

        log_variance = None
        if self.log_variance_head is not None:
            log_variance = self.log_variance_head(fused_feature.transpose(1, 2)).transpose(1, 2)
            log_variance = log_variance * point_mask.unsqueeze(-1).to(log_variance.dtype)

        return PointCloudTransitionPrediction(
            delta_points=delta_points,
            next_points=next_points,
            global_feature=global_feature,
            action_feature=action_feature,
            point_feature=fused_feature,
            log_variance=log_variance,
        )

    @torch.no_grad()
    def rollout(
        self,
        initial_points: torch.Tensor,
        actions: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> List[PointCloudTransitionPrediction]:
        if actions.ndim != 3:
            raise ValueError(f"actions must have shape (B, T, A), got {tuple(actions.shape)}")
        current_points = initial_points
        predictions = []
        for timestep in range(actions.shape[1]):
            prediction = self(current_points, actions[:, timestep], point_mask=point_mask)
            predictions.append(prediction)
            current_points = prediction.next_points
        return predictions


def build_lightweight_world_model(
    point_dim: int,
    action_dim: int,
    global_feature_dim: int = 128,
    action_feature_dim: int = 128,
    local_feature_dim: int = 128,
    hidden_dim: int = 256,
    predict_log_variance: bool = False,
) -> LightweightPointFlowWorldModel:
    return LightweightPointFlowWorldModel(
        point_dim=point_dim,
        action_dim=action_dim,
        global_feature_dim=global_feature_dim,
        action_feature_dim=action_feature_dim,
        local_feature_dim=local_feature_dim,
        hidden_dim=hidden_dim,
        predict_log_variance=predict_log_variance,
    )


@dataclass
class TrajectoryPointCloudPrediction:
    object_future_points: torch.Tensor
    object_future_delta: torch.Tensor
    object_history_feature: torch.Tensor
    contact_history_feature: torch.Tensor
    future_contact_feature: torch.Tensor
    global_feature: torch.Tensor


def _default_point_mask(
    batch_size: int,
    num_points: int,
    device: torch.device,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None:
        return torch.ones((batch_size, num_points), dtype=torch.bool, device=device)
    if mask.shape != (batch_size, num_points):
        raise ValueError(f"Expected mask shape {(batch_size, num_points)}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def _expand_temporal_mask(mask: Optional[torch.Tensor], horizon: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask[:, None, :].expand(-1, horizon, -1)


def _masked_point_feature_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ValueError(f"tensor must have at least 3 dimensions, got {tuple(tensor.shape)}")

    if mask is None:
        return tensor.mean(dim=-2)

    if mask.ndim != tensor.ndim - 1:
        raise ValueError(
            f"mask must have shape matching tensor without the last feature dimension, got "
            f"tensor {tuple(tensor.shape)} and mask {tuple(mask.shape)}"
        )

    expanded_mask = mask.to(dtype=tensor.dtype, device=tensor.device).unsqueeze(-1)
    denom = expanded_mask.sum(dim=-2).clamp_min(1.0)
    return (tensor * expanded_mask).sum(dim=-2) / denom


def _build_point_flow_sequence_features(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 4:
        raise ValueError(f"sequence must have shape (B, T, N, C), got {tuple(sequence.shape)}")
    if sequence.shape[-1] < 3:
        raise ValueError("sequence feature dimension must be at least 3 to compute xyz flow.")

    zero_flow = torch.zeros_like(sequence[:, :1, :, :3])
    point_flow = sequence[:, 1:, :, :3] - sequence[:, :-1, :, :3]
    point_flow = torch.cat([zero_flow, point_flow], dim=1)
    return torch.cat([sequence, point_flow], dim=-1)


def _encode_point_trajectories(
    sequence: torch.Tensor,
    projection: nn.Module,
    temporal_encoder: RNNLayer,
) -> torch.Tensor:
    batch_size, horizon, num_points, feature_dim = sequence.shape
    projected = projection(sequence.reshape(batch_size * horizon * num_points, feature_dim))
    projected = projected.reshape(batch_size, horizon, num_points, -1)

    rnn_inputs = projected.permute(1, 0, 2, 3).reshape(horizon * batch_size * num_points, -1)
    hidden = torch.zeros(
        batch_size * num_points,
        temporal_encoder._recurrent_N,
        temporal_encoder.rnn.hidden_size,
        device=sequence.device,
        dtype=sequence.dtype,
    )
    masks = torch.ones(horizon * batch_size * num_points, device=sequence.device, dtype=sequence.dtype)
    encoded, _ = temporal_encoder(rnn_inputs, hidden, masks)
    encoded = encoded.reshape(horizon, batch_size, num_points, -1).permute(1, 0, 2, 3)
    return encoded[:, -1]


def _encode_framewise_global_sequence(pointnet: nn.Module, sequence: torch.Tensor) -> torch.Tensor:
    batch_size, horizon, num_points, feature_dim = sequence.shape
    encoded = pointnet(sequence.reshape(batch_size * horizon, num_points, feature_dim))
    return encoded.reshape(batch_size, horizon, -1)


def trajectory_world_model_loss(
    prediction: TrajectoryPointCloudPrediction,
    target_object_future_points: torch.Tensor,
    object_mask: Optional[torch.Tensor] = None,
    loss_type: str = "ordered_mse",
    feature_weight: float = 0.25,
) -> torch.Tensor:
    if target_object_future_points.shape != prediction.object_future_points.shape:
        raise ValueError(
            f"target_object_future_points shape {tuple(target_object_future_points.shape)} must match "
            f"prediction.object_future_points shape {tuple(prediction.object_future_points.shape)}"
        )

    object_time_mask = _expand_temporal_mask(object_mask, prediction.object_future_points.shape[1])

    if loss_type == "ordered_mse":
        object_xyz_loss = masked_mean(
            (prediction.object_future_points[..., :3] - target_object_future_points[..., :3]).pow(2),
            object_time_mask,
        )

        object_feature_loss = prediction.object_future_points.new_tensor(0.0)
        if prediction.object_future_points.shape[-1] > 3:
            object_feature_loss = masked_mean(
                (prediction.object_future_points[..., 3:] - target_object_future_points[..., 3:]).pow(2),
                object_time_mask,
            )

        return object_xyz_loss + feature_weight * object_feature_loss

    if loss_type == "chamfer":
        total_loss = prediction.object_future_points.new_tensor(0.0)
        horizon = prediction.object_future_points.shape[1]
        for timestep in range(horizon):
            total_loss = total_loss + chamfer_distance(
                prediction.object_future_points[:, timestep, :, :3],
                target_object_future_points[:, timestep, :, :3],
                source_mask=object_mask,
                target_mask=object_mask,
            )
        return total_loss / float(horizon)

    raise ValueError(f"Unsupported loss_type: {loss_type}")


class TrajectoryConditionedPointFlowWorldModel(nn.Module):
    """
    A lightweight world model for object/contact disentanglement.

    It consumes:
    - object point trajectories over a history window,
    - contact-surface point trajectories over the same history window,
    - future contact-surface states as a structured replacement for raw actions.

    It predicts future object point clouds over the requested horizon.
    Contact trajectories are only used as conditioning signals.
    """

    def __init__(
        self,
        object_point_dim: int,
        contact_point_dim: int,
        trajectory_feature_dim: int = 128,
        global_feature_dim: int = 128,
        local_feature_dim: int = 128,
        hidden_dim: int = 256,
        recurrent_layers: int = 1,
        use_orthogonal_rnn: bool = False,
    ):
        super().__init__()
        if object_point_dim < 3 or contact_point_dim < 3:
            raise ValueError("object_point_dim and contact_point_dim must both be at least 3.")

        self.object_point_dim = int(object_point_dim)
        self.contact_point_dim = int(contact_point_dim)
        self.trajectory_feature_dim = int(trajectory_feature_dim)
        self.global_feature_dim = int(global_feature_dim)
        self.local_feature_dim = int(local_feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.object_history_projection = _build_linear_mlp(
            [self.object_point_dim + 3, self.trajectory_feature_dim, self.trajectory_feature_dim],
            inactivated_output=False,
        )
        self.contact_history_projection = _build_linear_mlp(
            [self.contact_point_dim + 3, self.trajectory_feature_dim, self.trajectory_feature_dim],
            inactivated_output=False,
        )
        self.future_contact_projection = _build_linear_mlp(
            [self.contact_point_dim + 3, self.trajectory_feature_dim, self.trajectory_feature_dim],
            inactivated_output=False,
        )

        self.object_history_temporal_encoder = RNNLayer(
            self.trajectory_feature_dim,
            self.trajectory_feature_dim,
            recurrent_layers,
            use_orthogonal_rnn,
        )
        self.contact_history_temporal_encoder = RNNLayer(
            self.trajectory_feature_dim,
            self.trajectory_feature_dim,
            recurrent_layers,
            use_orthogonal_rnn,
        )
        self.future_contact_temporal_encoder = RNNLayer(
            self.trajectory_feature_dim,
            self.trajectory_feature_dim,
            recurrent_layers,
            use_orthogonal_rnn,
        )

        self.object_global_encoder = _build_global_encoder(self.object_point_dim, self.global_feature_dim)
        self.contact_global_encoder = _build_global_encoder(self.contact_point_dim, self.global_feature_dim)
        self.object_local_encoder = _build_conv_mlp(
            [self.object_point_dim, self.local_feature_dim, self.local_feature_dim],
            inactivated_output=False,
        )
        self.contact_local_encoder = _build_conv_mlp(
            [self.contact_point_dim, self.local_feature_dim, self.local_feature_dim],
            inactivated_output=False,
        )

        object_fusion_dim = (
            self.object_point_dim
            + self.local_feature_dim
            + self.trajectory_feature_dim
            + self.global_feature_dim * 3
            + self.trajectory_feature_dim
            + self.local_feature_dim
            + self.contact_point_dim
        )
        self.object_fusion_encoder = _build_conv_mlp(
            [object_fusion_dim, self.hidden_dim, self.hidden_dim],
            inactivated_output=False,
        )

        self.object_xyz_head = _build_conv_mlp([self.hidden_dim, self.hidden_dim, 3], inactivated_output=True)

        self.object_feature_head = None
        if self.object_point_dim > 3:
            self.object_feature_head = _build_conv_mlp(
                [self.hidden_dim, self.hidden_dim, self.object_point_dim - 3],
                inactivated_output=True,
            )

    def forward(
        self,
        object_history: torch.Tensor,
        contact_history: torch.Tensor,
        future_contact_points: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        contact_mask: Optional[torch.Tensor] = None,
    ) -> TrajectoryPointCloudPrediction:
        if object_history.ndim != 4 or contact_history.ndim != 4 or future_contact_points.ndim != 4:
            raise ValueError("All inputs must have shape (B, T, N, C).")
        if object_history.shape[-1] != self.object_point_dim:
            raise ValueError(
                f"Expected object_history feature dim {self.object_point_dim}, got {object_history.shape[-1]}"
            )
        if contact_history.shape[-1] != self.contact_point_dim:
            raise ValueError(
                f"Expected contact_history feature dim {self.contact_point_dim}, got {contact_history.shape[-1]}"
            )
        if future_contact_points.shape[-1] != self.contact_point_dim:
            raise ValueError(
                f"Expected future_contact_points feature dim {self.contact_point_dim}, got "
                f"{future_contact_points.shape[-1]}"
            )
        if object_history.shape[0] != contact_history.shape[0] or object_history.shape[0] != future_contact_points.shape[0]:
            raise ValueError("Batch size must match across object_history, contact_history, and future_contact_points.")
        if contact_history.shape[2] != future_contact_points.shape[2]:
            raise ValueError("contact_history and future_contact_points must use the same number of contact points.")

        batch_size, _, object_points, _ = object_history.shape
        _, _, contact_points, _ = contact_history.shape
        future_horizon = future_contact_points.shape[1]

        object_mask = _default_point_mask(batch_size, object_points, object_history.device, object_mask)
        contact_mask = _default_point_mask(batch_size, contact_points, contact_history.device, contact_mask)

        object_history_feature = _encode_point_trajectories(
            _build_point_flow_sequence_features(object_history),
            self.object_history_projection,
            self.object_history_temporal_encoder,
        )
        contact_history_feature = _encode_point_trajectories(
            _build_point_flow_sequence_features(contact_history),
            self.contact_history_projection,
            self.contact_history_temporal_encoder,
        )
        future_contact_feature = _encode_point_trajectories(
            _build_point_flow_sequence_features(future_contact_points),
            self.future_contact_projection,
            self.future_contact_temporal_encoder,
        )

        object_history_global = _encode_framewise_global_sequence(self.object_global_encoder, object_history).mean(dim=1)
        contact_history_global = _encode_framewise_global_sequence(self.contact_global_encoder, contact_history).mean(dim=1)
        future_contact_global = _encode_framewise_global_sequence(self.contact_global_encoder, future_contact_points)

        last_object_points = object_history[:, -1]

        object_last_local = self.object_local_encoder(last_object_points.transpose(1, 2)).transpose(1, 2)
        future_contact_local = self.contact_local_encoder(
            future_contact_points.reshape(batch_size * future_horizon, contact_points, self.contact_point_dim).transpose(1, 2)
        ).transpose(1, 2).reshape(batch_size, future_horizon, contact_points, self.local_feature_dim)

        future_contact_summary = _masked_point_feature_mean(future_contact_feature, contact_mask)
        future_contact_step_summary = _masked_point_feature_mean(
            torch.cat([future_contact_points, future_contact_local], dim=-1),
            _expand_temporal_mask(contact_mask, future_horizon),
        )

        object_decoder_inputs = torch.cat(
            [
                last_object_points[:, None, :, :].expand(-1, future_horizon, -1, -1),
                object_last_local[:, None, :, :].expand(-1, future_horizon, -1, -1),
                object_history_feature[:, None, :, :].expand(-1, future_horizon, -1, -1),
                object_history_global[:, None, None, :].expand(-1, future_horizon, object_points, -1),
                contact_history_global[:, None, None, :].expand(-1, future_horizon, object_points, -1),
                future_contact_global[:, :, None, :].expand(-1, -1, object_points, -1),
                future_contact_summary[:, None, None, :].expand(-1, future_horizon, object_points, -1),
                future_contact_step_summary[:, :, None, :].expand(-1, -1, object_points, -1),
            ],
            dim=-1,
        ).reshape(batch_size * future_horizon, object_points, -1)
        object_decoder_feature = self.object_fusion_encoder(object_decoder_inputs.transpose(1, 2)).transpose(1, 2)
        object_xyz_delta = self.object_xyz_head(object_decoder_feature.transpose(1, 2)).transpose(1, 2)
        object_xyz_delta = object_xyz_delta.reshape(batch_size, future_horizon, object_points, 3)
        object_xyz_delta = object_xyz_delta * object_mask[:, None, :, None].to(object_xyz_delta.dtype)

        object_future_points = last_object_points[:, None, :, :].expand(-1, future_horizon, -1, -1).clone()
        object_future_points[..., :3] = object_future_points[..., :3] + object_xyz_delta
        if self.object_feature_head is not None:
            object_feature_delta = self.object_feature_head(object_decoder_feature.transpose(1, 2)).transpose(1, 2)
            object_feature_delta = object_feature_delta.reshape(
                batch_size, future_horizon, object_points, self.object_point_dim - 3
            )
            object_future_points[..., 3:] = object_future_points[..., 3:] + object_feature_delta

        global_feature = torch.cat(
            [
                object_history_global,
                contact_history_global,
                future_contact_global.mean(dim=1),
                future_contact_summary,
            ],
            dim=-1,
        )

        return TrajectoryPointCloudPrediction(
            object_future_points=object_future_points,
            object_future_delta=object_xyz_delta,
            object_history_feature=object_history_feature,
            contact_history_feature=contact_history_feature,
            future_contact_feature=future_contact_feature,
            global_feature=global_feature,
        )


def build_trajectory_world_model(
    object_point_dim: int,
    contact_point_dim: int,
    trajectory_feature_dim: int = 128,
    global_feature_dim: int = 128,
    local_feature_dim: int = 128,
    hidden_dim: int = 256,
    recurrent_layers: int = 1,
    use_orthogonal_rnn: bool = False,
) -> TrajectoryConditionedPointFlowWorldModel:
    return TrajectoryConditionedPointFlowWorldModel(
        object_point_dim=object_point_dim,
        contact_point_dim=contact_point_dim,
        trajectory_feature_dim=trajectory_feature_dim,
        global_feature_dim=global_feature_dim,
        local_feature_dim=local_feature_dim,
        hidden_dim=hidden_dim,
        recurrent_layers=recurrent_layers,
        use_orthogonal_rnn=use_orthogonal_rnn,
    )


__all__ = [
    "PointCloudTransitionPrediction",
    "TrajectoryPointCloudPrediction",
    "LightweightPointFlowWorldModel",
    "TrajectoryConditionedPointFlowWorldModel",
    "build_lightweight_world_model",
    "build_trajectory_world_model",
    "masked_mean",
    "chamfer_distance",
    "world_model_loss",
    "trajectory_world_model_loss",
]
