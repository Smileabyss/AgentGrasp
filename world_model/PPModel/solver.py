from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import torch

from .model import PPModel, PointCloudState


LossType = Literal["ordered_mse", "chamfer"]


@dataclass
class PPSolverResult:
    q: torch.Tensor
    state: PointCloudState
    final_loss: float
    iterations: int
    converged: bool
    loss_history: List[float] = field(default_factory=list)
    target_points: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PPModelSolver:
    def __init__(self, model: PPModel):
        self.model = model

        lower_limits, upper_limits = self.model.pk_chain.get_joint_limits()
        self.lower_limits = self._sanitize_limit_tensor(lower_limits, fill_value=-torch.inf)
        self.upper_limits = self._sanitize_limit_tensor(upper_limits, fill_value=torch.inf)

    def _sanitize_limit_tensor(self, values, fill_value: float):
        tensor = torch.as_tensor(values, dtype=self.model.dtype, device=self.model.device)
        tensor = tensor.clone()
        tensor[~torch.isfinite(tensor)] = fill_value
        return tensor

    def _prepare_points(self, points, name: str):
        tensor = torch.as_tensor(points, dtype=self.model.dtype, device=self.model.device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (N, 3) or (B, N, 3), got {tuple(tensor.shape)}")
        return tensor

    def _prepare_optional_weights(self, weights, target_shape, name: str):
        if weights is None:
            return None
        tensor = torch.as_tensor(weights, dtype=self.model.dtype, device=self.model.device)
        if tensor.ndim == len(target_shape) - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and target_shape[0] > 1:
            tensor = tensor.expand(target_shape[0], -1)
        if tuple(tensor.shape) != tuple(target_shape):
            raise ValueError(f"{name} must have shape {tuple(target_shape)}, got {tuple(tensor.shape)}")
        return tensor

    def _prepare_initial_q(self, initial_q, batch_size: int):
        if initial_q is None:
            q = torch.zeros((batch_size, self.model.dof), dtype=self.model.dtype, device=self.model.device)
            return self._project_q(q)

        q = self.model._prepare_joint_tensor(initial_q, "initial_q").detach().clone()
        if q.shape[0] == 1 and batch_size > 1:
            q = q.expand(batch_size, -1).clone()
        if q.shape[0] != batch_size:
            raise ValueError(f"initial_q batch size {q.shape[0]} must match target batch size {batch_size}")
        return self._project_q(q)

    def _project_q(self, q: torch.Tensor):
        lower = self.lower_limits.unsqueeze(0).expand_as(q)
        upper = self.upper_limits.unsqueeze(0).expand_as(q)
        projected = torch.minimum(torch.maximum(q, lower), upper)
        projected = torch.where(torch.isfinite(projected), projected, q)
        return projected

    def _compute_point_loss(
        self,
        pred_points: torch.Tensor,
        target_points: torch.Tensor,
        loss_type: LossType,
        point_weights: Optional[torch.Tensor] = None,
    ):
        if loss_type == "ordered_mse":
            if pred_points.shape != target_points.shape:
                raise ValueError(
                    f"ordered_mse requires matching point cloud shapes, got {tuple(pred_points.shape)} "
                    f"and {tuple(target_points.shape)}"
                )
            sq_error = (pred_points - target_points).pow(2).sum(dim=-1)
            if point_weights is not None:
                denom = point_weights.sum(dim=-1).clamp_min(1e-8)
                return ((sq_error * point_weights).sum(dim=-1) / denom).mean()
            return sq_error.mean()

        if loss_type == "chamfer":
            dist = torch.cdist(pred_points, target_points, p=2).pow(2)
            pred_to_target = dist.min(dim=-1)[0]
            target_to_pred = dist.min(dim=-2)[0]
            if point_weights is not None:
                denom = point_weights.sum(dim=-1).clamp_min(1e-8)
                pred_term = ((pred_to_target * point_weights).sum(dim=-1) / denom).mean()
            else:
                pred_term = pred_to_target.mean()
            return pred_term + target_to_pred.mean()

        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def _expand_index_tensor(self, tensor, batch_size: int, name: str):
        tensor = torch.as_tensor(tensor, dtype=torch.long, device=self.model.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"{name} must have shape (E,) or (B, E), got {tuple(tensor.shape)}")
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, -1)
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {tensor.shape[0]} must match batch size {batch_size}")
        return tensor

    def _expand_value_tensor(self, tensor, batch_size: int, edge_count: int, name: str, default_value: float):
        if tensor is None:
            return torch.full(
                (batch_size, edge_count),
                float(default_value),
                dtype=self.model.dtype,
                device=self.model.device,
            )

        value = torch.as_tensor(tensor, dtype=self.model.dtype, device=self.model.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        if tuple(value.shape) != (batch_size, edge_count):
            raise ValueError(f"{name} must have shape {(batch_size, edge_count)}, got {tuple(value.shape)}")
        return value

    def _compute_contact_graph_loss(self, pred_points: torch.Tensor, object_points: torch.Tensor, contact_graph: Dict[str, Any]):
        batch_size = pred_points.shape[0]
        hand_indices = self._expand_index_tensor(contact_graph["hand_indices"], batch_size, "hand_indices")
        object_indices = self._expand_index_tensor(contact_graph["object_indices"], batch_size, "object_indices")
        if hand_indices.shape != object_indices.shape:
            raise ValueError("hand_indices and object_indices must have the same shape")

        edge_count = hand_indices.shape[1]
        target_distance = self._expand_value_tensor(
            contact_graph.get("target_distance"),
            batch_size,
            edge_count,
            "target_distance",
            0.0,
        )
        edge_weight = self._expand_value_tensor(
            contact_graph.get("weight"),
            batch_size,
            edge_count,
            "weight",
            1.0,
        )

        if hand_indices.min() < 0 or hand_indices.max() >= pred_points.shape[1]:
            raise ValueError("hand_indices contains out-of-range point indices")
        if object_indices.min() < 0 or object_indices.max() >= object_points.shape[1]:
            raise ValueError("object_indices contains out-of-range point indices")

        batch_index = torch.arange(batch_size, device=self.model.device).unsqueeze(-1)
        hand_points = pred_points[batch_index, hand_indices]
        object_contact_points = object_points[batch_index, object_indices]
        actual_distance = torch.linalg.norm(hand_points - object_contact_points, dim=-1)

        sq_error = (actual_distance - target_distance).pow(2)
        denom = edge_weight.sum(dim=-1).clamp_min(1e-8)
        return ((sq_error * edge_weight).sum(dim=-1) / denom).mean()

    def _estimate_point_from_distances(
        self,
        reference_points: torch.Tensor,
        distances: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        if reference_points.ndim != 2 or reference_points.shape[-1] != 3:
            raise ValueError("reference_points must have shape (M, 3)")
        if distances.ndim != 1 or distances.shape[0] != reference_points.shape[0]:
            raise ValueError("distances must have shape (M,)")

        if weights is None:
            weights = torch.ones_like(distances)
        else:
            weights = torch.as_tensor(weights, dtype=self.model.dtype, device=self.model.device)
            if weights.shape != distances.shape:
                raise ValueError("weights must match distances shape")

        weights = weights.clamp_min(1e-8)
        points = reference_points.unsqueeze(-1)
        distances_sq = distances.pow(2)
        weight_column = weights[:, None, None]

        point_transpose = points.transpose(1, 2)
        point_outer = points @ point_transpose
        point_norm_sq = (point_transpose @ points).squeeze(-1).squeeze(-1)

        eye = torch.eye(3, dtype=self.model.dtype, device=self.model.device)
        normalization = weights.sum().clamp_min(1e-8)
        residual = (point_norm_sq - distances_sq)[:, None, None]
        a_term = (weight_column * (residual * points)).sum(dim=0) / normalization
        b_term = (weight_column * (-2 * point_outer - residual * eye.unsqueeze(0))).sum(dim=0) / normalization
        mean_point = (weight_column * points).sum(dim=0) / normalization
        f_term = a_term + b_term @ mean_point + 2 * mean_point @ mean_point.transpose(0, 1) @ mean_point
        h_term = -2 * (weight_column * point_outer).sum(dim=0) / normalization + 2 * mean_point @ mean_point.transpose(0, 1)

        solution = -torch.linalg.pinv(h_term) @ f_term
        estimate = solution + mean_point
        return estimate[:, 0]

    def reconstruct_points_from_contact_map(self, object_points, contact_map, contact_weights=None):
        object_points = self._prepare_points(object_points, "object_points")
        contact_map = torch.as_tensor(contact_map, dtype=self.model.dtype, device=self.model.device)
        if contact_map.ndim == 2:
            contact_map = contact_map.unsqueeze(0)
        if contact_map.ndim != 3:
            raise ValueError(
                f"contact_map must have shape (N_hand, N_object) or (B, N_hand, N_object), got {tuple(contact_map.shape)}"
            )

        if object_points.shape[0] == 1 and contact_map.shape[0] > 1:
            object_points = object_points.expand(contact_map.shape[0], -1, -1)
        if object_points.shape[0] != contact_map.shape[0]:
            raise ValueError("object_points and contact_map batch size must match")
        if object_points.shape[1] != contact_map.shape[2]:
            raise ValueError("contact_map last dimension must equal the number of object points")

        if contact_weights is not None:
            contact_weights = torch.as_tensor(contact_weights, dtype=self.model.dtype, device=self.model.device)
            if contact_weights.ndim == 2:
                contact_weights = contact_weights.unsqueeze(0)
            if contact_weights.shape != contact_map.shape:
                raise ValueError("contact_weights must have the same shape as contact_map")

        batch_points = []
        for batch_idx in range(contact_map.shape[0]):
            estimated_points = []
            for hand_idx in range(contact_map.shape[1]):
                weights = None if contact_weights is None else contact_weights[batch_idx, hand_idx]
                estimated_points.append(
                    self._estimate_point_from_distances(
                        object_points[batch_idx],
                        contact_map[batch_idx, hand_idx],
                        weights=weights,
                    )
                )
            batch_points.append(torch.stack(estimated_points, dim=0))
        return torch.stack(batch_points, dim=0)

    def _run_q_optimization(
        self,
        initial_q,
        *,
        target_points=None,
        point_weights=None,
        loss_type: LossType = "ordered_mse",
        object_points=None,
        contact_graph=None,
        iterations: int = 200,
        lr: float = 5e-2,
        optimizer_name: Literal["adam", "lbfgs"] = "adam",
        point_loss_weight: float = 1.0,
        contact_loss_weight: float = 1.0,
        regularization_weight: float = 1e-4,
        q_prior=None,
        tolerance: float = 1e-6,
        clamp_to_limits: bool = True,
    ):
        prepared_target_points = None
        prepared_object_points = None
        batch_size = None

        if target_points is not None:
            prepared_target_points = self._prepare_points(target_points, "target_points")
            batch_size = prepared_target_points.shape[0]
            if point_weights is not None:
                point_weights = self._prepare_optional_weights(
                    point_weights,
                    prepared_target_points.shape[:2],
                    "point_weights",
                )

        if batch_size is None and initial_q is not None:
            batch_size = self.model._prepare_joint_tensor(initial_q, "initial_q").shape[0]

        if object_points is not None:
            prepared_object_points = self._prepare_points(object_points, "object_points")
            if batch_size is None:
                batch_size = prepared_object_points.shape[0]
            elif prepared_object_points.shape[0] == 1 and batch_size > 1:
                prepared_object_points = prepared_object_points.expand(batch_size, -1, -1)
            elif prepared_object_points.shape[0] != batch_size:
                raise ValueError("object_points batch size must match target_points batch size")

        if batch_size is None:
            raise ValueError("At least one optimization target must be provided")

        q = self._prepare_initial_q(initial_q, batch_size).requires_grad_(True)
        q_prior = self._prepare_initial_q(q_prior, batch_size) if q_prior is not None else q.detach().clone()

        if optimizer_name == "adam":
            optimizer = torch.optim.Adam([q], lr=lr)
        elif optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS([q], lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        else:
            raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

        best_loss = float("inf")
        best_q = q.detach().clone()
        loss_history: List[float] = []
        converged = False
        previous_loss = None

        def compute_total_loss():
            pred_points = self.model.get_point_cloud(q)
            pred_points = pred_points.unsqueeze(0) if pred_points.ndim == 2 else pred_points

            loss = torch.zeros((), dtype=self.model.dtype, device=self.model.device)
            if prepared_target_points is not None:
                loss = loss + point_loss_weight * self._compute_point_loss(
                    pred_points,
                    prepared_target_points,
                    loss_type=loss_type,
                    point_weights=point_weights,
                )
            if prepared_object_points is not None and contact_graph is not None:
                loss = loss + contact_loss_weight * self._compute_contact_graph_loss(
                    pred_points,
                    prepared_object_points,
                    contact_graph,
                )
            if regularization_weight > 0:
                loss = loss + regularization_weight * (q - q_prior).pow(2).mean()
            return loss, pred_points

        for _ in range(iterations):
            if optimizer_name == "lbfgs":
                cache = {}

                def closure():
                    optimizer.zero_grad()
                    total_loss, pred_points = compute_total_loss()
                    total_loss.backward()
                    cache["loss"] = total_loss.detach()
                    cache["pred_points"] = pred_points.detach()
                    return total_loss

                optimizer.step(closure)
                total_loss = cache["loss"]
            else:
                optimizer.zero_grad()
                total_loss, _ = compute_total_loss()
                total_loss.backward()
                optimizer.step()

            if clamp_to_limits:
                with torch.no_grad():
                    q.copy_(self._project_q(q))

            with torch.no_grad():
                total_loss, _ = compute_total_loss()
            loss_value = float(total_loss.item())
            loss_history.append(loss_value)

            if loss_value < best_loss:
                best_loss = loss_value
                best_q = q.detach().clone()

            if previous_loss is not None and abs(previous_loss - loss_value) < tolerance:
                converged = True
                break
            previous_loss = loss_value

        final_q = best_q
        final_state = self.model.get_state(final_q)
        return PPSolverResult(
            q=final_q[0] if final_q.shape[0] == 1 else final_q,
            state=final_state,
            final_loss=best_loss,
            iterations=len(loss_history),
            converged=converged,
            loss_history=loss_history,
            target_points=prepared_target_points[0] if prepared_target_points is not None and prepared_target_points.shape[0] == 1 else prepared_target_points,
        )

    def solve_q_from_point_cloud(
        self,
        target_points,
        initial_q=None,
        *,
        point_weights=None,
        loss_type: LossType = "ordered_mse",
        iterations: int = 200,
        lr: float = 5e-2,
        optimizer_name: Literal["adam", "lbfgs"] = "adam",
        regularization_weight: float = 1e-4,
        q_prior=None,
        tolerance: float = 1e-6,
        clamp_to_limits: bool = True,
    ):
        return self._run_q_optimization(
            initial_q=initial_q,
            target_points=target_points,
            point_weights=point_weights,
            loss_type=loss_type,
            iterations=iterations,
            lr=lr,
            optimizer_name=optimizer_name,
            regularization_weight=regularization_weight,
            q_prior=q_prior,
            tolerance=tolerance,
            clamp_to_limits=clamp_to_limits,
        )

    def optimize_q_from_contact_map(
        self,
        object_points,
        contact_map,
        initial_q=None,
        *,
        contact_weights=None,
        dense_map_loss_type: LossType = "ordered_mse",
        iterations: int = 250,
        lr: float = 5e-2,
        optimizer_name: Literal["adam", "lbfgs"] = "adam",
        point_loss_weight: float = 1.0,
        contact_loss_weight: float = 1.0,
        regularization_weight: float = 1e-4,
        q_prior=None,
        tolerance: float = 1e-6,
        clamp_to_limits: bool = True,
    ):
        if isinstance(contact_map, dict):
            result = self._run_q_optimization(
                initial_q=initial_q,
                object_points=object_points,
                contact_graph=contact_map,
                iterations=iterations,
                lr=lr,
                optimizer_name=optimizer_name,
                contact_loss_weight=contact_loss_weight,
                regularization_weight=regularization_weight,
                q_prior=initial_q if q_prior is None else q_prior,
                tolerance=tolerance,
                clamp_to_limits=clamp_to_limits,
            )
            result.metadata["contact_mode"] = "sparse_graph"
            return result

        reconstructed_points = self.reconstruct_points_from_contact_map(
            object_points=object_points,
            contact_map=contact_map,
            contact_weights=contact_weights,
        )
        result = self._run_q_optimization(
            initial_q=initial_q,
            target_points=reconstructed_points,
            loss_type=dense_map_loss_type,
            iterations=iterations,
            lr=lr,
            optimizer_name=optimizer_name,
            point_loss_weight=point_loss_weight,
            regularization_weight=regularization_weight,
            q_prior=q_prior,
            tolerance=tolerance,
            clamp_to_limits=clamp_to_limits,
        )
        result.metadata["contact_mode"] = "dense_distance_map"
        return result
