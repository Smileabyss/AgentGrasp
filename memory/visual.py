#!/usr/bin/env python3
"""Play an OakInk MANO hand trajectory with its object in a 3-D viewer.

Controls: Space pause/play, R restart, Left/Right step one frame, Q close.
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "datasets/oakink_human_trajectories.npz"
DEFAULT_MESH_ROOT = Path("/home/cyrus/ManipTrans/data/OakInk-v2/object_preview/align_ds")
DEFAULT_MANO_MODEL = Path("/home/cyrus/ManipTrans/data/mano_v1_2/models/MANO_RIGHT.pkl")


def patch_legacy_numpy_aliases() -> None:
    """Let chumpy-backed MANO pickles load with NumPy 1.24+."""
    import builtins

    aliases = {"unicode": str}
    for name in ("bool", "int", "float", "complex", "object", "unicode", "str"):
        if name not in np.__dict__:
            setattr(np, name, aliases.get(name, getattr(builtins, name, str)))


def load_object_mesh(object_id: str, mesh_root: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mesh_path = mesh_root / object_id / "scan.ply"
    if not mesh_path.exists():
        return None, None
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        return vertices, faces
    except (ImportError, OSError, ValueError) as error:
        print(f"Could not load object mesh {mesh_path}: {error}")
        return None, None


def import_smplx():
    """Import smplx, including a sibling Conda base install if present."""
    try:
        return importlib.import_module("smplx")
    except ImportError:
        # On this workstation smplx is installed in the Conda base Python,
        # while chumpy (needed to read MANO_RIGHT.pkl) is in the active env.
        prefix = Path(sys.prefix).resolve()
        conda_root = prefix.parent.parent if prefix.parent.name == "envs" else prefix
        for site_packages in sorted((conda_root / "lib").glob("python*/site-packages")):
            if (site_packages / "smplx" / "__init__.py").is_file():
                sys.path.insert(0, str(site_packages))
                importlib.invalidate_caches()
                return importlib.import_module("smplx")
        raise RuntimeError(
            "smplx is required to render MANO. Install smplx in this Python environment."
        )


def load_mano_meshes(
    data: np.lib.npyio.NpzFile,
    index: int,
    wrist_world: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the OakInk right MANO mesh for every resampled frame."""
    source_path = Path(str(data["source_path"][index]))
    if not source_path.exists():
        raise FileNotFoundError(f"OakInk source annotation is missing: {source_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MANO model is missing: {model_path}")

    source_begin = int(data["source_frame_range_count"][index, 0])
    with source_path.open("rb") as handle:
        annotation = pickle.load(handle)
    betas = np.asarray(annotation["raw_mano"][source_begin]["rh__betas"], dtype=np.float32)
    betas = betas.reshape(-1, 10)[0]
    return decode_mano_meshes(data["mano_pose_coeffs"][index], betas, wrist_world, model_path)


def decode_mano_meshes(
    mano_pose_coeffs: np.ndarray,
    betas: np.ndarray,
    wrist_world: np.ndarray,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode OakInk MANO poses, treating ``rh__tsl`` as the wrist position.

    OakInk stores ``rh__tsl`` at the MANO wrist joint.  The ``transl`` input
    of smplx is the MANO template/root origin, which is several centimetres
    away from that joint and rotates with the root pose.  Passing ``rh__tsl``
    directly as ``transl`` therefore shifts the whole hand.  Decode at zero
    translation, then apply the per-frame offset that places joint 0 at the
    recorded wrist position.
    """
    # The 16 OakInk quaternions are root orientation followed by 15 MANO joints,
    # stored as wxyz. smplx expects axis-angle vectors.
    mano_quaternions = np.asarray(mano_pose_coeffs, dtype=np.float64).reshape(-1, 16, 4)
    mano_rotvecs = Rotation.from_quat(mano_quaternions[..., [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec()
    mano_rotvecs = mano_rotvecs.reshape(-1, 16, 3)

    try:
        patch_legacy_numpy_aliases()
        smplx = import_smplx()
        model = smplx.create(
            model_path=str(model_path),
            model_type="mano",
            is_rhand=True,
            use_pca=False,
            flat_hand_mean=True,
        )
    except ModuleNotFoundError as error:
        if error.name == "chumpy":
            raise RuntimeError(
                "The MANO pickle needs chumpy. Run python -m memory.visual in the DemoGrasp Python environment."
            ) from error
        raise

    import torch

    model.eval()
    device = torch.device("cpu")
    vertices = []
    with torch.no_grad():
        for start in range(0, len(wrist_world), 128):
            stop = min(start + 128, len(wrist_world))
            output = model(
                global_orient=torch.as_tensor(mano_rotvecs[start:stop, 0], dtype=torch.float32, device=device),
                hand_pose=torch.as_tensor(
                    mano_rotvecs[start:stop, 1:].reshape(-1, 45), dtype=torch.float32, device=device
                ),
                betas=torch.as_tensor(
                    np.repeat(np.asarray(betas, dtype=np.float32)[None], stop - start, axis=0),
                    dtype=torch.float32, device=device,
                ),
                transl=torch.zeros((stop - start, 3), dtype=torch.float32, device=device),
            )
            vertices_zero = output.vertices
            wrist_zero = output.joints[:, 0]
            target_wrist = torch.as_tensor(
                wrist_world[start:stop, :3], dtype=torch.float32, device=device
            )
            vertices.append(
                (vertices_zero + (target_wrist - wrist_zero)[:, None, :]).cpu().numpy()
            )
    vertices = np.concatenate(vertices, axis=0).astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, faces


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_raw_trajectory(
    data: np.lib.npyio.NpzFile,
    index: int,
    model_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and decode all valid source annotation frames for one trajectory."""
    source_path = Path(str(data["source_path"][index]))
    if not source_path.exists():
        raise FileNotFoundError(f"OakInk source annotation is missing: {source_path}")
    source_begin, source_end = map(int, data["source_frame_range_count"][index, :2])
    object_id = str(data["object_id"][index])
    with source_path.open("rb") as handle:
        annotation = pickle.load(handle)

    raw_mano = annotation["raw_mano"]
    object_transforms = annotation["obj_transf"][object_id]
    frame_ids = np.asarray(
        [
            frame_id for frame_id in annotation["mocap_frame_id_list"]
            if source_begin <= frame_id <= source_end
            and frame_id in raw_mano and frame_id in object_transforms
        ],
        dtype=np.int64,
    )
    if frame_ids.size == 0:
        raise ValueError(f"No valid raw frames for {object_id} in {source_path}")

    positions = []
    poses = []
    object_poses = []
    for frame_id in frame_ids:
        frame = raw_mano[int(frame_id)]
        positions.append(_to_numpy(frame["rh__tsl"]).reshape(-1, 3)[0].astype(np.float64))
        poses.append(_to_numpy(frame["rh__pose_coeffs"]).reshape(-1).astype(np.float64))
        object_poses.append(_to_numpy(object_transforms[int(frame_id)]).reshape(4, 4).astype(np.float64))

    positions = np.stack(positions)
    poses = np.stack(poses)
    mano_quaternions = poses.reshape(-1, 16, 4)
    root_quaternions = mano_quaternions[:, 0]
    wrist_quaternions = root_quaternions[:, [1, 2, 3, 0]]
    wrist_world = np.concatenate((positions, wrist_quaternions), axis=-1)
    object_poses = np.stack(object_poses)
    betas = _to_numpy(raw_mano[int(frame_ids[0])]["rh__betas"]).reshape(-1, 10)[0]
    mano_vertices_world, mano_faces = decode_mano_meshes(poses, betas, wrist_world, model_path)
    return frame_ids, wrist_world, object_poses, mano_vertices_world, mano_faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--index", type=int, default=0, help="trajectory row in the NPZ")
    parser.add_argument("--raw", action="store_true", help="play every valid source annotation frame")
    parser.add_argument("--source-frame", type=int, help="initial source frame when using --raw")
    parser.add_argument(
        "--source-window", type=int,
        help="show this many raw frames centered on --source-frame (useful for inspecting contact)",
    )
    parser.add_argument("--frame", choices=("object", "world"), default=None)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--axis-length", type=float, default=0.06)
    parser.add_argument("--mesh-root", type=Path, default=DEFAULT_MESH_ROOT)
    parser.add_argument("--mano-model", type=Path, default=DEFAULT_MANO_MODEL)
    parser.add_argument("--no-mesh", action="store_true", help="hide the optional OakInk object mesh")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    count = data["human_trajectory"].shape[0]
    if args.index < 0 or args.index >= count:
        parser.error(f"--index must be in [0, {count - 1}]")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.source_frame is not None and not args.raw:
        parser.error("--source-frame requires --raw")
    if args.source_window is not None and (not args.raw or args.source_window < 1):
        parser.error("--source-window requires --raw and must be positive")
    if args.source_window is not None and args.source_frame is None:
        parser.error("--source-window requires --source-frame")
    frame_mode = args.frame or ("world" if args.raw else "object")

    if args.raw:
        frame_labels, wrist_world, object_pose, mano_vertices_world, mano_faces = load_raw_trajectory(
            data, args.index, args.mano_model
        )
    else:
        trajectory = np.asarray(data["human_trajectory"][args.index], dtype=np.float64)
        object_pose = np.asarray(data["object_pose_world"][args.index], dtype=np.float64)
        wrist_world = np.asarray(data["wrist_pose_world"][args.index], dtype=np.float64)
        frame_labels = np.arange(1, len(trajectory) + 1)
        mano_vertices_world, mano_faces = load_mano_meshes(data, args.index, wrist_world, args.mano_model)

    if args.source_window is not None:
        target = int(np.argmin(np.abs(frame_labels - args.source_frame)))
        half_window = args.source_window // 2
        first = max(0, target - half_window)
        last = min(len(frame_labels), first + args.source_window)
        first = max(0, last - args.source_window)
        frame_labels = frame_labels[first:last]
        wrist_world = wrist_world[first:last]
        object_pose = object_pose[first:last]
        mano_vertices_world = mano_vertices_world[first:last]

    wrist_position_world = wrist_world[:, :3]
    wrist_rotation_world = Rotation.from_quat(wrist_world[:, [4, 5, 6, 3]]).as_matrix()
    if frame_mode == "object":
        object_rotation_world = object_pose[:, :3, :3]
        object_position_world = object_pose[:, :3, 3]
        object_rotation_transpose = object_rotation_world.transpose(0, 2, 1)
        wrist_position = np.einsum(
            "tij,tj->ti", object_rotation_transpose, wrist_position_world - object_position_world
        )
        wrist_rotation = np.einsum("tij,tjk->tik", object_rotation_transpose, wrist_rotation_world)
        object_position = np.zeros((len(wrist_world), 3))
        object_rotation = np.repeat(np.eye(3)[None], len(wrist_world), axis=0)
        title_frame = "object frame"
        mano_vertices = np.einsum(
            "tij,tvj->tvi", object_rotation_transpose,
            mano_vertices_world - object_position_world[:, None, :],
        )
    else:
        wrist_position = wrist_position_world
        object_position = object_pose[:, :3, 3]
        object_rotation = object_pose[:, :3, :3]
        wrist_rotation = wrist_rotation_world
        title_frame = "world frame"
        mano_vertices = mano_vertices_world

    start_frame = 0
    if args.source_frame is not None:
        start_frame = int(np.argmin(np.abs(frame_labels - args.source_frame)))

    object_id = str(data["object_id"][args.index])
    primitive = str(data["primitive"][args.index])
    source = str(data["source_path"][args.index])
    source_range = np.asarray(data["source_frame_range_count"][args.index])
    print(f"trajectory {args.index}/{count - 1}: object={object_id}, primitive={primitive}")
    print(f"source: {source}")
    print(f"selected source frames: {source_range[0]}..{source_range[1]} ({source_range[2]} frames)")
    mode = "raw source" if args.raw else "resampled"
    print(f"displaying {len(wrist_position)} {mode} MANO hand meshes in {title_frame}")
    if args.raw:
        print(f"display source frame range: {frame_labels[0]}..{frame_labels[-1]}")

    mesh_vertices, mesh_faces = (None, None) if args.no_mesh else load_object_mesh(object_id, args.mesh_root)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"OakInk MANO hand playback | {object_id} | {primitive} | {title_frame}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    if mesh_vertices is not None:
        first_object_vertices = object_rotation[0] @ mesh_vertices.T
        first_object_vertices = first_object_vertices.T + object_position[0]
        object_surface = Poly3DCollection(
            first_object_vertices[mesh_faces], facecolor="gray", edgecolor="none", alpha=0.42,
        )
        ax.add_collection3d(object_surface)
    else:
        object_vertices = None
        object_surface = None
        ax.scatter([object_position[0, 0]], [object_position[0, 1]], [object_position[0, 2]],
                   marker="s", s=50, color="black", label="object origin")

    ax.plot(wrist_position[:, 0], wrist_position[:, 1], wrist_position[:, 2],
            color="tab:blue", alpha=0.65, linewidth=1.5, label="wrist path")
    wrist_point, = ax.plot([], [], [], "o", color="tab:red", markersize=7, label="wrist")
    hand_mesh = Poly3DCollection(
        mano_vertices[0, mano_faces], facecolor="tab:orange", edgecolor=(0.3, 0.16, 0.08, 0.35),
        linewidth=0.08, alpha=0.9,
    )
    ax.add_collection3d(hand_mesh)
    path_trace, = ax.plot([], [], [], color="tab:red", linewidth=2.0, alpha=0.8, label="played path")
    triad = [
        ax.plot([], [], [], color=color, linewidth=2.5)[0]
        for color in ("tab:red", "tab:green", "tab:blue")
    ]
    object_axes = [
        ax.plot([], [], [], color=color, linewidth=1.1, alpha=0.5)[0]
        for color in ("tab:red", "tab:green", "tab:blue")
    ]
    frame_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes)

    cloud = np.concatenate((wrist_position, object_position, mano_vertices.reshape(-1, 3)), axis=0)
    lower = cloud.min(axis=0)
    upper = cloud.max(axis=0)
    if mesh_vertices is not None:
        object_radius = float(np.linalg.norm(mesh_vertices, axis=-1).max())
        lower = np.minimum(lower, object_position.min(axis=0) - object_radius)
        upper = np.maximum(upper, object_position.max(axis=0) + object_radius)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0, 0.05) * 1.2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    handles, labels = ax.get_legend_handles_labels()
    if object_surface is not None:
        handles.append(Patch(facecolor="gray", alpha=0.42, label="object surface"))
        labels.append("object surface")
    handles.append(Patch(facecolor="tab:orange", label="MANO right hand"))
    labels.append("MANO right hand")
    ax.legend(handles, labels, loc="upper right")

    state = {"frame": start_frame, "playing": True}

    def set_line(line, start: np.ndarray, end: np.ndarray) -> None:
        line.set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

    def update(frame: int):
        state["frame"] = frame % len(wrist_position)
        current = state["frame"]
        point = wrist_position[current]
        hand_mesh.set_verts(mano_vertices[current, mano_faces])
        wrist_point.set_data_3d([point[0]], [point[1]], [point[2]])
        path_trace.set_data_3d(wrist_position[: current + 1, 0],
                               wrist_position[: current + 1, 1],
                               wrist_position[: current + 1, 2])
        for axis_index, line in enumerate(triad):
            endpoint = point + args.axis_length * wrist_rotation[current, :, axis_index]
            set_line(line, point, endpoint)
        obj_origin = object_position[current]
        for axis_index, line in enumerate(object_axes):
            endpoint = obj_origin + args.axis_length * 0.8 * object_rotation[current, :, axis_index]
            set_line(line, obj_origin, endpoint)
        if object_surface is not None:
            current_object_vertices = object_rotation[current] @ mesh_vertices.T
            current_object_vertices = current_object_vertices.T + object_position[current]
            object_surface.set_verts(current_object_vertices[mesh_faces])
        frame_text.set_text(
            f"source frame {frame_labels[current]} ({current + 1}/{len(wrist_position)})  |  "
            "Space: pause/play  R: restart  ←/→: step"
        )
        return [hand_mesh, wrist_point, path_trace, *triad, *object_axes, frame_text] + (
            [object_surface] if object_surface is not None else []
        )

    def on_key(event):
        if event.key == " ":
            state["playing"] = not state["playing"]
            if state["playing"]:
                animation.event_source.start()
            else:
                animation.event_source.stop()
        elif event.key in ("r", "R"):
            state["frame"] = start_frame
            update(start_frame)
            fig.canvas.draw_idle()
        elif event.key == "right":
            state["playing"] = False
            animation.event_source.stop()
            update(state["frame"] + 1)
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["playing"] = False
            animation.event_source.stop()
            update(state["frame"] - 1)
            fig.canvas.draw_idle()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    animation = FuncAnimation(
        fig, update, frames=range(start_frame, start_frame + len(wrist_position)),
        interval=1000.0 / args.fps, blit=False, cache_frame_data=False,
    )
    update(start_frame)
    plt.show()


if __name__ == "__main__":
    main()
