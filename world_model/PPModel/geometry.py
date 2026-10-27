from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        return trimesh.util.concatenate(
            tuple(
                trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces)
                for geom in scene_or_mesh.geometry.values()
            )
        )
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    raise TypeError(f"Unsupported trimesh object: {type(scene_or_mesh)!r}")


def parse_origin(element: ET.Element):
    origin = element.find("origin")
    xyz = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def apply_transform(mesh: trimesh.Trimesh, translation: np.ndarray, rotation: np.ndarray):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry: ET.Element, translation: np.ndarray, rotation: np.ndarray):
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ", dtype=np.float64)
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        mesh = trimesh.creation.icosphere(radius=float(geometry.attrib["radius"]))
    elif geometry.tag.endswith("cylinder"):
        mesh = trimesh.creation.cylinder(
            radius=float(geometry.attrib["radius"]),
            height=float(geometry.attrib["length"]),
        )
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def resolve_mesh_path(mesh_filename: str, urdf_dir: Path, package_map: Optional[Dict[str, str]] = None):
    if mesh_filename.startswith("file://"):
        return Path(mesh_filename[len("file://"):]).expanduser().resolve()

    if mesh_filename.startswith("package://"):
        package_spec = mesh_filename[len("package://"):]
        package_name, _, relative_path = package_spec.partition("/")
        if not package_name or not relative_path:
            raise ValueError(f"Malformed package URI: {mesh_filename}")

        if package_map and package_name in package_map:
            return (Path(package_map[package_name]) / relative_path).expanduser().resolve()

        search_roots = [urdf_dir, *urdf_dir.parents]
        for root in search_roots:
            candidate = root / package_name / relative_path
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Cannot resolve {mesh_filename}. Provide package_map={{'{package_name}': '/path/to/package'}}."
        )

    mesh_path = Path(mesh_filename).expanduser()
    if mesh_path.is_absolute():
        return mesh_path.resolve()
    return (urdf_dir / mesh_path).resolve()


def load_link_geometries(
    urdf_path: str,
    link_names: Iterable[str],
    geometry_type: str = "visual",
    package_map: Optional[Dict[str, str]] = None,
):
    if geometry_type not in {"visual", "collision"}:
        raise ValueError(f"geometry_type must be 'visual' or 'collision', got {geometry_type!r}")

    urdf_path = Path(urdf_path).expanduser().resolve()
    urdf_dir = urdf_path.parent
    root = ET.parse(urdf_path).getroot()
    requested_links = set(link_names)

    link_geometries = {}
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        if link_name not in requested_links:
            continue

        link_meshes = []
        for visual_or_collision in link.findall(f".//{geometry_type}"):
            geometry_container = visual_or_collision.find("geometry")
            if geometry_container is None or len(geometry_container) == 0:
                continue

            geometry = geometry_container[0]
            xyz, rotation = parse_origin(visual_or_collision)

            if geometry.tag.endswith("mesh"):
                mesh_filename = geometry.attrib["filename"]
                mesh_path = resolve_mesh_path(mesh_filename, urdf_dir, package_map=package_map)
                mesh = as_mesh(trimesh.load(str(mesh_path)))
                if mesh is None:
                    continue
                scale = np.fromstring(geometry.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
                mesh.apply_scale(scale)
                mesh = apply_transform(mesh, xyz, rotation)
            else:
                mesh = create_primitive_mesh(geometry, xyz, rotation)
                scale = np.fromstring(geometry.attrib.get("scale", "1 1 1"), sep=" ", dtype=np.float64)
                mesh.apply_scale(scale)

            link_meshes.append(mesh)

        if not link_meshes:
            continue
        if len(link_meshes) == 1:
            link_geometries[link_name] = link_meshes[0]
        else:
            link_geometries[link_name] = as_mesh(trimesh.Scene(link_meshes))

    return link_geometries
