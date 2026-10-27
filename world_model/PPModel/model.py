from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch
import pytorch_kinematics as pk

from .geometry import load_link_geometries
from .kinematics import compute_link_jacobians


@dataclass
class PointCloudState:
    points: torch.Tensor
    velocities: Optional[torch.Tensor]
    link_indices: torch.Tensor
    link_names: List[str]
    joint_names: List[str]


class PPModel:
    @staticmethod
    def _resolve_urdf_path(urdf_path: str) -> Path:
        input_path = Path(urdf_path).expanduser()
        if input_path.is_absolute():
            resolved_path = input_path.resolve()
            if resolved_path.exists():
                return resolved_path
            raise FileNotFoundError(f"URDF file not found: {resolved_path}")

        project_root = Path(__file__).resolve().parent.parent
        candidate_paths = [
            Path.cwd() / input_path,
            project_root / input_path,
        ]

        for candidate_path in candidate_paths:
            resolved_path = candidate_path.resolve()
            if resolved_path.exists():
                return resolved_path

        tried_paths = ", ".join(str(candidate.resolve()) for candidate in candidate_paths)
        raise FileNotFoundError(f"URDF file not found: {urdf_path}. Tried: {tried_paths}")

    @staticmethod
    def _normalize_root_link_names(root_link_names: Optional[str | Sequence[str]]):
        if root_link_names is None:
            return None
        if isinstance(root_link_names, str) and root_link_names.strip().lower() in {"", "none", "null"}:
            return None
        return root_link_names

    def __init__(
        self,
        urdf_path: str,
        points_per_link: int = 128,
        geometry_type: str = "visual",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        root_link_names: Optional[str | Sequence[str]] = None,
        include_links: Optional[Sequence[str]] = None,
        exclude_links: Optional[Sequence[str]] = None,
        package_map: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        enable_viewer: bool = False,
        viewer_host: str = "127.0.0.1",
        viewer_port: int = 8080,
        viewer_point_size: float = 0.002,
        viewer_show_mesh: bool = False,
        viewer_show_velocity: bool = False,
        viewer_velocity_dt: float = 0.05,
    ):
        self.urdf_path = str(self._resolve_urdf_path(urdf_path))
        self.device = torch.device(device)
        self.dtype = dtype
        self.points_per_link = int(points_per_link)
        self.geometry_type = geometry_type
        self.root_link_names = self._normalize_root_link_names(root_link_names)
        self.package_map = package_map or {}
        self.seed = seed
        self.enable_viewer = enable_viewer
        self.viewer_host = viewer_host
        self.viewer_port = int(viewer_port)
        self.viewer_point_size = float(viewer_point_size)
        self.viewer_show_mesh = viewer_show_mesh
        self.viewer_show_velocity = viewer_show_velocity
        self.viewer_velocity_dt = float(viewer_velocity_dt)
        self._viser_module = None
        self._viewer_server = None
        self._viewer_points_handle = None
        self._viewer_velocity_handle = None

        urdf_text = Path(self.urdf_path).read_text()
        self.pk_chain = pk.build_chain_from_urdf(urdf_text).to(dtype=self.dtype, device=self.device)
        self.joint_names = list(self.pk_chain.get_joint_parameter_names())
        self.dof = len(self.joint_names)
        self.all_link_names = list(self.pk_chain.get_link_names())
        self._urdf_parent_to_children = self._parse_urdf_link_tree(urdf_text)

        requested_links = list(self.all_link_names)
        if self.root_link_names is not None:
            requested_links = self._collect_subtree_link_names(self.root_link_names)
        if include_links is not None:
            requested_links = [name for name in requested_links if name in set(include_links)]
        if exclude_links is not None:
            requested_links = [name for name in requested_links if name not in set(exclude_links)]

        self.meshes = load_link_geometries(
            urdf_path=self.urdf_path,
            link_names=requested_links,
            geometry_type=self.geometry_type,
            package_map=self.package_map,
        )
        self.link_names = [name for name in requested_links if name in self.meshes]
        if not self.link_names:
            raise ValueError(
                f"No link meshes were loaded from {self.urdf_path} using geometry_type={self.geometry_type!r}"
            )

        self.local_points_by_link = {}
        self._point_slices = {}
        self._link_indices = []

        point_offset = 0
        for link_index, link_name in enumerate(self.link_names):
            sampled = self._sample_link_points(self.meshes[link_name], self.points_per_link, link_index)
            sampled_tensor = torch.tensor(sampled, dtype=self.dtype, device=self.device)
            self.local_points_by_link[link_name] = sampled_tensor
            self._point_slices[link_name] = slice(point_offset, point_offset + sampled_tensor.shape[0])
            self._link_indices.extend([link_index] * sampled_tensor.shape[0])
            point_offset += sampled_tensor.shape[0]

        self.link_indices = torch.tensor(self._link_indices, dtype=torch.long, device=self.device)
        self.num_points = point_offset

        if self.enable_viewer:
            self._setup_viewer()

    @staticmethod
    def _parse_urdf_link_tree(urdf_text: str):
        root = ET.fromstring(urdf_text)
        parent_to_children = {}

        for joint in root.findall("joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue

            parent_link = parent.attrib.get("link")
            child_link = child.attrib.get("link")
            if not parent_link or not child_link:
                continue

            parent_to_children.setdefault(parent_link, []).append(child_link)

        return parent_to_children

    def _compute_frame_status(self, q):
        q = self._prepare_joint_tensor(q, "q")
        return q, self.pk_chain.forward_kinematics(q)

    def _sample_link_points(self, mesh, num_points: int, link_index: int):
        if self.seed is None:
            return mesh.sample(num_points)

        np_state = np.random.get_state()
        np.random.seed(self.seed + link_index)
        try:
            return mesh.sample(num_points)
        finally:
            np.random.set_state(np_state)

    def _setup_viewer(self):
        if self._viewer_server is not None:
            return

        try:
            self._viser_module = importlib.import_module("viser")
        except ImportError as exc:
            raise ImportError(
                "enable_viewer=True requires the 'viser' package. Install it or disable the viewer."
            ) from exc

        self._viewer_server = self._viser_module.ViserServer(
            host=self.viewer_host,
            port=self.viewer_port,
        )

    def _update_viewer(self, state: PointCloudState, q):
        if not self.enable_viewer:
            return
        self._setup_viewer()

        points = state.points
        velocities = state.velocities
        if points.ndim == 3:
            points = points[0]
            if velocities is not None:
                velocities = velocities[0]

        if self.viewer_show_mesh:
            for link_name, mesh in self.get_transformed_meshes(q).items():
                self._viewer_server.scene.add_mesh_simple(
                    f"mesh/{link_name}",
                    mesh.vertices,
                    mesh.faces,
                    color=(160, 160, 160),
                    opacity=0.45,
                )

        self._viewer_points_handle = self._viewer_server.scene.add_point_cloud(
            "points/current",
            points.detach().cpu().numpy(),
            point_size=self.viewer_point_size,
            point_shape="circle",
            colors=(0, 120, 255),
        )

        if self.viewer_show_velocity and velocities is not None:
            preview_points = points + velocities * self.viewer_velocity_dt
            self._viewer_velocity_handle = self._viewer_server.scene.add_point_cloud(
                "points/velocity_preview",
                preview_points.detach().cpu().numpy(),
                point_size=self.viewer_point_size,
                point_shape="circle",
                colors=(255, 80, 80),
            )
            if hasattr(self._viewer_velocity_handle, "visible"):
                self._viewer_velocity_handle.visible = True
        elif self._viewer_velocity_handle is not None:
            if hasattr(self._viewer_velocity_handle, "visible"):
                self._viewer_velocity_handle.visible = False
            elif hasattr(self._viewer_velocity_handle, "remove"):
                self._viewer_velocity_handle.remove()
                self._viewer_velocity_handle = None

    def get_joint_names(self):
        return list(self.joint_names)

    def get_link_names(self):
        return list(self.link_names)

    def _normalize_link_name_sequence(self, link_names, arg_name: str):
        if isinstance(link_names, str):
            names = [link_names]
        else:
            names = list(link_names)

        if not names:
            raise ValueError(f"{arg_name} must contain at least one link name.")
        return names

    def _collect_subtree_link_names(self, root_link_names: str | Sequence[str]):
        root_names = self._normalize_link_name_sequence(root_link_names, "root_link_names")
        unknown_roots = [name for name in root_names if name not in self.all_link_names]
        if unknown_roots:
            raise ValueError(
                f"Unknown root link name(s): {unknown_roots}. Available links include: {self.all_link_names}"
            )

        selected = set()
        pending = list(root_names)
        while pending:
            current = pending.pop()
            if current in selected:
                continue
            selected.add(current)
            pending.extend(self._urdf_parent_to_children.get(current, []))

        return [link_name for link_name in self.all_link_names if link_name in selected]

    def to(self, device: str | torch.device):
        target_device = torch.device(device)
        if target_device == self.device:
            return self

        self.device = target_device
        self.pk_chain = self.pk_chain.to(dtype=self.dtype, device=self.device)
        self.local_points_by_link = {
            link_name: points.to(self.device)
            for link_name, points in self.local_points_by_link.items()
        }
        self.link_indices = self.link_indices.to(self.device)
        return self

    def _prepare_joint_tensor(self, q, name: str):
        tensor = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.dof:
            raise ValueError(f"{name} must have shape ({self.dof},) or (B, {self.dof}), got {tuple(tensor.shape)}")
        return tensor

    def get_link_local_points(self):
        return {
            link_name: points.detach().clone()
            for link_name, points in self.local_points_by_link.items()
        }

    def get_transformed_meshes(self, q):
        q, frame_status = self._compute_frame_status(q)
        transformed_meshes = {}

        for link_name in self.link_names:
            transform = frame_status[link_name].get_matrix()[0].detach().cpu().numpy()
            transformed_meshes[link_name] = self.meshes[link_name].copy().apply_transform(transform)

        return transformed_meshes

    def get_state(self, q, qdot=None):
        q = self._prepare_joint_tensor(q, "q")
        compute_velocity = qdot is not None
        if qdot is None:
            qdot_tensor = None
        else:
            qdot_tensor = self._prepare_joint_tensor(qdot, "qdot")
            if qdot_tensor.shape != q.shape:
                raise ValueError(f"qdot shape {tuple(qdot_tensor.shape)} must match q shape {tuple(q.shape)}")

        frame_status = self.pk_chain.forward_kinematics(q)
        jacobians = None
        if compute_velocity:
            jacobians = compute_link_jacobians(self.pk_chain, q, frame_status, self.link_names)

        batch_points = []
        batch_velocities = [] if compute_velocity else None
        for link_name in self.link_names:
            local_points = self.local_points_by_link[link_name]
            transform = frame_status[link_name].get_matrix()
            rotation = transform[:, :3, :3]
            translation = transform[:, :3, 3]

            world_points = local_points.unsqueeze(0) @ rotation.transpose(1, 2) + translation.unsqueeze(1)

            batch_points.append(world_points)
            if compute_velocity:
                twist = torch.einsum("bij,bj->bi", jacobians[link_name], qdot_tensor)
                linear_velocity = twist[:, :3]
                angular_velocity = twist[:, 3:]
                offsets = world_points - translation.unsqueeze(1)
                point_velocities = linear_velocity.unsqueeze(1) + torch.cross(
                    angular_velocity.unsqueeze(1).expand_as(offsets),
                    offsets,
                    dim=-1,
                )
                batch_velocities.append(point_velocities)

        points = torch.cat(batch_points, dim=1)
        velocities = None
        if compute_velocity:
            velocities = torch.cat(batch_velocities, dim=1)

        if points.shape[0] == 1:
            points = points[0]
            if velocities is not None:
                velocities = velocities[0]

        state = PointCloudState(
            points=points,
            velocities=velocities,
            link_indices=self.link_indices.detach().clone(),
            link_names=self.get_link_names(),
            joint_names=self.get_joint_names(),
        )
        self._update_viewer(state, q)
        return state

    def get_point_cloud(self, q):
        return self.get_state(q=q, qdot=None).points

    def get_point_velocities(self, q, qdot):
        return self.get_state(q=q, qdot=qdot).velocities

    def create_solver(self):
        from .solver import PPModelSolver

        return PPModelSolver(self)
