from __future__ import annotations

from typing import List, Optional, Union

import torch
from scipy.spatial import ConvexHull, Delaunay, QhullError

from .model import PPModel


def _prepare_point_tensor(
    points,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(points, device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] < 3:
        raise ValueError(f"{name} must have shape (N, C) or (B, N, C) with C >= 3, got {tuple(tensor.shape)}")
    return tensor


def _expand_batch(tensor: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, -1, -1)
    raise ValueError(f"{name} batch size {tensor.shape[0]} must be 1 or {batch_size}")


def _farthest_point_sample_single(points_xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    num_points = points_xyz.shape[0]
    if num_samples >= num_points:
        return torch.arange(num_points, device=points_xyz.device, dtype=torch.long)

    selected = torch.zeros(num_samples, device=points_xyz.device, dtype=torch.long)
    min_dist = torch.full((num_points,), float("inf"), device=points_xyz.device, dtype=points_xyz.dtype)
    farthest = 0

    for sample_idx in range(num_samples):
        selected[sample_idx] = farthest
        centroid = points_xyz[farthest].unsqueeze(0)
        dist = ((points_xyz - centroid) ** 2).sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        farthest = torch.argmax(min_dist).item()

    return selected


def _compute_penetration_mask(hand_points_xyz: torch.Tensor, object_points_xyz: torch.Tensor) -> torch.Tensor:
    if object_points_xyz.shape[0] < 4:
        return torch.zeros(hand_points_xyz.shape[0], device=hand_points_xyz.device, dtype=torch.bool)

    try:
        object_cpu = object_points_xyz.detach().cpu().numpy()
        hull = ConvexHull(object_cpu)
        delaunay = Delaunay(object_cpu[hull.vertices])
        inside = delaunay.find_simplex(hand_points_xyz.detach().cpu().numpy()) >= 0
        return torch.as_tensor(inside, device=hand_points_xyz.device, dtype=torch.bool)
    except (QhullError, ValueError):
        return torch.zeros(hand_points_xyz.shape[0], device=hand_points_xyz.device, dtype=torch.bool)


def _downsample_if_needed(points: torch.Tensor, num_contact_points: int) -> torch.Tensor:
    if points.shape[0] <= num_contact_points:
        return points
    keep = _farthest_point_sample_single(points[:, :3], num_contact_points)
    return points[keep]


class PPContactSurfaceExtractor:
    def __init__(
        self,
        model: PPModel,
        num_contact_points: int = 64,
        outside_distance_threshold: float = 0.01,
    ):
        if num_contact_points <= 0:
            raise ValueError("num_contact_points must be positive.")
        if outside_distance_threshold < 0:
            raise ValueError("outside_distance_threshold must be non-negative.")

        self.model = model
        self.num_contact_points = int(num_contact_points)
        self.outside_distance_threshold = float(outside_distance_threshold)

    def get_contact_surface_points(
        self,
        q,
        object_points,
        *,
        num_contact_points: Optional[int] = None,
        outside_distance_threshold: Optional[float] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        state = self.model.get_state(q)
        hand_points = state.points
        if hand_points.ndim == 2:
            hand_points = hand_points.unsqueeze(0)

        object_points = _prepare_point_tensor(
            object_points,
            device=hand_points.device,
            dtype=hand_points.dtype,
            name="object_points",
        )
        object_points = _expand_batch(object_points, hand_points.shape[0], "object_points")

        num_contact_points = self.num_contact_points if num_contact_points is None else int(num_contact_points)
        if num_contact_points <= 0:
            raise ValueError("num_contact_points must be positive.")
        outside_distance_threshold = (
            self.outside_distance_threshold
            if outside_distance_threshold is None
            else float(outside_distance_threshold)
        )
        if outside_distance_threshold < 0:
            raise ValueError("outside_distance_threshold must be non-negative.")

        hand_xyz = hand_points[..., :3]
        object_xyz = object_points[..., :3]
        distance_matrix = torch.cdist(hand_xyz, object_xyz, p=2)
        min_distances = distance_matrix.min(dim=-1).values

        selected_hand_points = []
        for batch_idx in range(hand_points.shape[0]):
            batch_hand_points = hand_points[batch_idx]
            batch_distances = min_distances[batch_idx]
            penetration_mask = _compute_penetration_mask(batch_hand_points[:, :3], object_xyz[batch_idx])

            if penetration_mask.any():
                outside_near_mask = (~penetration_mask) & (batch_distances < outside_distance_threshold)
                candidate_mask = penetration_mask | outside_near_mask
                candidate_points = batch_hand_points[candidate_mask]
                if candidate_points.shape[0] == 0:
                    sorted_indices = torch.argsort(batch_distances)[:num_contact_points]
                    candidate_points = batch_hand_points[sorted_indices]
                selected_points = _downsample_if_needed(candidate_points, num_contact_points)
            else:
                sorted_indices = torch.argsort(batch_distances)[:num_contact_points]
                selected_points = batch_hand_points[sorted_indices]

            selected_hand_points.append(selected_points)

        if hand_points.shape[0] == 1:
            return selected_hand_points[0]

        if all(points.shape == selected_hand_points[0].shape for points in selected_hand_points):
            return torch.stack(selected_hand_points, dim=0)
        return selected_hand_points


def extract_contact_surface_points(
    model: PPModel,
    q,
    object_points,
    *,
    num_contact_points: int = 64,
    outside_distance_threshold: float = 0.01,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    extractor = PPContactSurfaceExtractor(
        model=model,
        num_contact_points=num_contact_points,
        outside_distance_threshold=outside_distance_threshold,
    )
    return extractor.get_contact_surface_points(
        q=q,
        object_points=object_points,
        outside_distance_threshold=outside_distance_threshold,
    )


__all__ = [
    "PPContactSurfaceExtractor",
    "extract_contact_surface_points",
]
