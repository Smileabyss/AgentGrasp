"""Train and evaluate a continuous temporal AutoEncoder for grasp trajectories.

The trajectory representation is exactly the one used by
``memory.train_trajectory_vqvae``:

    wrist position in the initial object frame (3)
    wrist rotation in the initial object frame as 6D rotation (6)
    raw active hand joint positions (6 for Inspire, 20 for Wuji)

The model input dimension is inferred from the dataset, so Wuji trajectories
use 29 features per frame while Inspire trajectories continue to use 15.

This script deliberately has no vector quantizer.  It is a control experiment
for deciding whether the large rotation error comes from the VQ stage.

Examples
--------
Train a continuous temporal AutoEncoder::

    python -m memory.train_trajectory_ae train \
        --dataset datasets/inspire_grasp.npz \
        --output-dir runs/trajectory_ae \
        --epochs 200 --num-tokens 10 --latent-dim 64

Evaluate its reconstruction and rotation error::

    python -m memory.train_trajectory_ae evaluate \
        --dataset datasets/inspire_grasp.npz \
        --checkpoint runs/trajectory_ae/best.pt \
        --output runs/trajectory_ae/evaluation.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from memory.train_trajectory_vqvae import (
    prepare_trajectories,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_matrix,
    quaternion_to_rotation6d,
    rotation6d_to_matrix,
)


class TemporalAutoEncoder(nn.Module):
    """Convolutional temporal encoder/decoder without quantization."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        num_tokens: int = 10,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(num_tokens),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=5, padding=2),
        )

    def encode(self, trajectory: Tensor) -> Tensor:
        return self.encoder(trajectory.transpose(1, 2)).transpose(1, 2)

    def decode(self, latent: Tensor) -> Tensor:
        upsampled = torch.nn.functional.interpolate(
            latent.transpose(1, 2),
            size=self.sequence_length,
            mode="linear",
            align_corners=False,
        )
        return self.decoder(upsampled).transpose(1, 2)

    def forward(self, trajectory: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encode(trajectory)
        return self.decode(latent), latent


def _load_data(path: str) -> Tensor:
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    return prepare_trajectories(data)


def _sliding_trajectory_windows(trajectories: Tensor, traj_length: int) -> Tensor:
    """Return every stride-one window as a separate trajectory sample."""
    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected trajectories with shape [N, T, D], got {tuple(trajectories.shape)}"
        )
    source_length = trajectories.shape[1]
    if traj_length < 1 or traj_length > source_length:
        raise ValueError(
            f"traj_length must be between 1 and source length {source_length}, got {traj_length}"
        )
    windows = trajectories.unfold(1, traj_length, 1).permute(0, 1, 3, 2)
    return windows.reshape(-1, traj_length, trajectories.shape[-1]).contiguous()


def _load_hand_joint_names(path: str, hand_joint_count: int) -> list[str]:
    names = None
    with np.load(path, allow_pickle=False) as loaded:
        if "palm_joint_names" in loaded.files:
            names = [str(name) for name in loaded["palm_joint_names"].tolist()]
    metadata_path = Path(path).with_suffix(".json")
    if names is None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata_names = metadata.get("palm_joint_names")
        if isinstance(metadata_names, list):
            names = [str(name) for name in metadata_names]
    if names is None:
        names = [f"hand_joint_{index}" for index in range(hand_joint_count)]
    if len(names) != hand_joint_count:
        raise ValueError(
            f"Dataset has {hand_joint_count} hand joint channels but "
            f"{len(names)} joint names"
        )
    return names


def _load_world_data(path: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load world-frame palm poses, object poses, joints, and object clouds."""
    with np.load(path, allow_pickle=False) as loaded:
        wrist = torch.from_numpy(np.asarray(loaded["wrist_trajectory"], dtype=np.float32))
        object_pose = torch.from_numpy(np.asarray(loaded["object_pose_world"], dtype=np.float32))
        joints = torch.from_numpy(
            np.asarray(loaded["palm_joint_position_trajectory"], dtype=np.float32)
        )
        pointcloud = torch.from_numpy(
            np.asarray(loaded["object_pointcloud"], dtype=np.float32)
        )
    if wrist.ndim != 3 or wrist.shape[-1] != 7:
        raise ValueError("wrist_trajectory must have shape [N, T, 7]")
    if object_pose.ndim != 2 or object_pose.shape[-1] != 7:
        raise ValueError("object_pose_world must have shape [N, 7]")
    if joints.ndim != 3 or joints.shape[:2] != wrist.shape[:2]:
        raise ValueError("palm_joint_position_trajectory must match wrist trajectory shape")
    if pointcloud.ndim != 3 or pointcloud.shape[0] != wrist.shape[0] or pointcloud.shape[-1] != 3:
        raise ValueError("object_pointcloud must have shape [N, P, 3]")
    return wrist, object_pose, joints, pointcloud


def _sample_virtual_frames(
    wrist: Tensor,
    object_pose: Tensor,
    joints: Tensor,
    pointcloud: Tensor,
    samples: int,
) -> Tensor:
    """Transform world trajectories into frames anchored in object geometry."""
    if samples < 1:
        raise ValueError("virtual-frame sample count must be positive")
    batch = wrist.shape[0]
    device = wrist.device
    dtype = wrist.dtype
    object_rotation = quaternion_to_matrix(object_pose[:, 3:7])
    wrist_position_object = torch.einsum(
        "bij,btj->bti", object_rotation.transpose(-1, -2),
        wrist[..., :3] - object_pose[:, None, :3],
    )
    wrist_rotation_object = torch.matmul(
        object_rotation.transpose(-1, -2)[:, None], quaternion_to_matrix(wrist[..., 3:7])
    )
    point_index = torch.randint(pointcloud.shape[1], (batch, samples), device=device)
    surface = torch.gather(
        pointcloud, 1, point_index[..., None].expand(-1, -1, 3)
    )
    center = pointcloud.mean(dim=1, keepdim=True)
    # Use only physically meaningful surface anchors. Interior anchors would
    # create virtual frames at locations where the robot cannot contact the
    # object and can make the latent encode artificial geometry.
    origins = surface
    radial = torch.nn.functional.normalize(
        surface - center, dim=-1
    )
    reference = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).expand_as(radial)
    parallel = radial[..., 2].abs() > 0.9
    reference = torch.where(parallel[..., None], torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype), reference)
    x_axis = torch.nn.functional.normalize(torch.cross(reference, radial, dim=-1), dim=-1)
    y_axis = torch.cross(radial, x_axis, dim=-1)
    yaw = torch.rand((batch, samples), device=device, dtype=dtype) * (2.0 * math.pi)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    frame_x = cosine[..., None] * x_axis - sine[..., None] * y_axis
    frame_y = sine[..., None] * x_axis + cosine[..., None] * y_axis
    # Matrix columns are the virtual frame's X/Y/Z axes in object coordinates.
    frame_rotation = torch.stack((frame_x, frame_y, radial), -1)
    position = torch.einsum(
        "bsij,bstj->bsti", frame_rotation.transpose(-1, -2),
        wrist_position_object[:, None] - origins[:, :, None, :],
    )
    relative_rotation = torch.matmul(frame_rotation.transpose(-1, -2)[:, :, None], wrist_rotation_object[:, None])
    rotation6d = relative_rotation[..., :, :2].transpose(-1, -2).reshape(
        batch, samples, wrist.shape[1], 6
    )
    joints = joints[:, None].expand(-1, samples, -1, -1)
    features = torch.cat((position, rotation6d, joints), dim=-1)
    return features.reshape(batch * samples, wrist.shape[1], 9 + joints.shape[-1])


def _build_model(config: Dict) -> TemporalAutoEncoder:
    return TemporalAutoEncoder(
        input_dim=int(config["input_dim"]),
        sequence_length=int(config["sequence_length"]),
        num_tokens=int(config["num_tokens"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )


def _rotation_matrix_loss(target: Tensor, reconstruction: Tensor) -> Tensor:
    """Differentiable rotation loss used in addition to 6D feature MSE."""

    target_matrix = rotation6d_to_matrix(target[..., 3:9])
    reconstruction_matrix = rotation6d_to_matrix(reconstruction[..., 3:9])
    return (target_matrix - reconstruction_matrix).square().mean()


def _loss(
    target: Tensor,
    reconstruction: Tensor,
    rotation_matrix_loss_weight: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    feature_mse = (reconstruction - target).square().mean()
    rotation_mse = _rotation_matrix_loss(target, reconstruction)
    total = feature_mse + rotation_matrix_loss_weight * rotation_mse
    return total, feature_mse, rotation_mse


def train(args: argparse.Namespace) -> None:
    traj_length = (
        args.sequence_length if args.traj_length is None else args.traj_length
    )
    if traj_length < 1:
        raise ValueError("traj_length must be positive")
    if args.virtual_frame_samples > 0:
        world_wrist, object_pose, hand_joints, object_pointcloud = _load_world_data(args.dataset)
        trajectories = None
        sample_count = world_wrist.shape[0]
        hand_joint_count = hand_joints.shape[-1]
    else:
        trajectories = _load_data(args.dataset)
        world_wrist = object_pose = hand_joints = object_pointcloud = None
        sample_count = trajectories.shape[0]
        if trajectories.ndim != 3 or trajectories.shape[-1] <= 9:
            raise ValueError(
                "Expected trajectories with shape [N, T, 9 + hand_joint_count], "
                f"got {tuple(trajectories.shape)}"
            )
        hand_joint_count = trajectories.shape[-1] - 9
    input_dim = 9 + hand_joint_count
    hand_joint_names = _load_hand_joint_names(args.dataset, hand_joint_count)
    sequence_length = world_wrist.shape[1] if trajectories is None else trajectories.shape[1]
    if sequence_length != args.sequence_length:
        raise ValueError(
            f"Expected {args.sequence_length} trajectory steps, got {sequence_length}"
        )
    if traj_length > sequence_length:
        raise ValueError(
            f"traj_length ({traj_length}) cannot exceed source trajectory length ({sequence_length})"
        )
    if sample_count < 2:
        raise ValueError("At least two trajectories are required for a train/validation split")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    permutation = torch.randperm(sample_count)
    split = max(1, int(sample_count * (1.0 - args.validation_fraction)))
    train_indices = permutation[:split]
    valid_indices = permutation[split:]
    if valid_indices.shape[0] == 0:
        valid_indices = train_indices[:1]

    model = TemporalAutoEncoder(
        input_dim=input_dim,
        sequence_length=traj_length,
        num_tokens=args.num_tokens,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_indices, batch_size=args.batch_size, shuffle=True, drop_last=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name
    best_validation = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals = []
        for batch_indices in train_loader:
            batch_indices = batch_indices.long()
            if trajectories is None:
                batch = _sample_virtual_frames(
                    world_wrist[batch_indices].to(args.device),
                    object_pose[batch_indices].to(args.device),
                    hand_joints[batch_indices].to(args.device),
                    object_pointcloud[batch_indices].to(args.device),
                    args.virtual_frame_samples,
                )
            else:
                batch = trajectories[batch_indices].to(args.device)
            batch = _sliding_trajectory_windows(batch, traj_length)
            reconstruction, _ = model(batch)
            total, _, _ = _loss(batch, reconstruction, args.rotation_matrix_loss_weight)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_totals.append(float(total.detach().cpu()))

        model.eval()
        validation_totals = []
        with torch.no_grad():
            for start in range(0, valid_indices.shape[0], args.batch_size):
                indices = valid_indices[start : start + args.batch_size].long()
                if trajectories is None:
                    batch = _sample_virtual_frames(
                        world_wrist[indices].to(args.device),
                        object_pose[indices].to(args.device),
                        hand_joints[indices].to(args.device),
                        object_pointcloud[indices].to(args.device),
                        args.virtual_frame_samples,
                    )
                else:
                    batch = trajectories[indices].to(args.device)
                batch = _sliding_trajectory_windows(batch, traj_length)
                reconstruction, _ = model(batch)
                total, _, _ = _loss(batch, reconstruction, args.rotation_matrix_loss_weight)
                validation_totals.append(float(total.cpu()))
        train_loss = float(np.mean(train_totals))
        validation_loss = float(np.mean(validation_totals))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d}/{args.epochs}: "
                f"train={train_loss:.6f} val={validation_loss:.6f}"
            )
        if validation_loss < best_validation:
            best_validation = validation_loss
            checkpoint = {
                "model": model.state_dict(),
                "config": {
                    "input_dim": input_dim,
                    "hand_joint_count": hand_joint_count,
                    "hand_joint_names": hand_joint_names,
                    "sequence_length": traj_length,
                    "source_sequence_length": sequence_length,
                    "window_stride": 1,
                    "num_tokens": args.num_tokens,
                    "latent_dim": args.latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "rotation_matrix_loss_weight": args.rotation_matrix_loss_weight,
                    "feature_order": [
                        "wrist_position_object",
                        "wrist_rotation6d_object",
                        *[f"hand_joint:{name}" for name in hand_joint_names],
                    ],
                    "normalization": "none; wrist position is relative to a sampled object surface frame",
                    "coordinate_frame": "virtual_object_pointcloud" if args.virtual_frame_samples > 0 else "initial_object",
                    "virtual_frame_samples": args.virtual_frame_samples,
                },
                "epoch": epoch,
                "validation_loss": validation_loss,
            }
            torch.save(checkpoint, checkpoint_path)

    print(f"Saved continuous AutoEncoder checkpoint to {checkpoint_path}")


def evaluate(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if checkpoint["config"].get("coordinate_frame") == "virtual_object_pointcloud":
        wrist, object_pose, joints, pointcloud = _load_world_data(args.dataset)
        samples = max(1, int(checkpoint["config"].get("virtual_frame_samples", 1)))
        trajectories = _sample_virtual_frames(
            wrist.to(args.device), object_pose.to(args.device), joints.to(args.device),
            pointcloud.to(args.device),
            samples,
        ).cpu()
    else:
        trajectories = _load_data(args.dataset)
    traj_length = int(checkpoint["config"]["sequence_length"])
    trajectories = _sliding_trajectory_windows(trajectories, traj_length)
    model = _build_model(checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    reconstructions = []
    latents = []
    with torch.no_grad():
        for start in range(0, trajectories.shape[0], args.batch_size):
            batch = trajectories[start : start + args.batch_size].to(args.device)
            reconstruction, latent = model(batch)
            reconstructions.append(reconstruction.cpu())
            latents.append(latent.cpu())
    reconstruction = torch.cat(reconstructions, dim=0)
    latent = torch.cat(latents, dim=0)
    error = reconstruction - trajectories

    metrics = {
        "samples": int(trajectories.shape[0]),
        "trajectory_shape": list(trajectories.shape),
        "latent_shape": list(latent.shape),
        "reconstruction_mse": float(error.square().mean()),
        "reconstruction_rmse": float(error.square().mean().sqrt()),
        "rotation_matrix_mse": float(_rotation_matrix_loss(trajectories, reconstruction)),
        "wrist_position_object_mse": float(error[..., :3].square().mean()),
        "wrist_position_object_rmse": float(error[..., :3].square().mean().sqrt()),
        "wrist_rotation6d_object_mse": float(error[..., 3:9].square().mean()),
        "wrist_rotation6d_object_rmse": float(error[..., 3:9].square().mean().sqrt()),
        "raw_hand_joints_mse": float(error[..., 9:].square().mean()),
        "raw_hand_joints_rmse": float(error[..., 9:].square().mean().sqrt()),
        "wrist_position_final_rmse": float(error[:, -1, :3].square().mean().sqrt()),
        "raw_hand_joints_final_rmse": float(error[:, -1, 9:].square().mean().sqrt()),
    }
    true_rotation = rotation6d_to_matrix(trajectories[..., 3:9])
    predicted_rotation = rotation6d_to_matrix(reconstruction[..., 3:9])
    relative_rotation = torch.matmul(true_rotation.transpose(-1, -2), predicted_rotation)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    metrics["wrist_rotation_geodesic_mean_deg"] = float(angle.mean() * 180.0 / np.pi)
    metrics["wrist_rotation_geodesic_median_deg"] = float(angle.median() * 180.0 / np.pi)
    metrics["wrist_rotation_geodesic_95th_deg"] = float(
        torch.quantile(angle, 0.95) * 180.0 / np.pi
    )

    print(json.dumps(metrics, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved evaluation metrics to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--output-dir", default="runs/trajectory_ae")
    train_parser.add_argument("--checkpoint-name", default="best.pt")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--sequence-length", type=int, default=20)
    train_parser.add_argument(
        "--traj_length", "--traj-length", dest="traj_length", type=int, default=None,
        help="trajectory window length; defaults to --sequence-length",
    )
    train_parser.add_argument("--num-tokens", type=int, default=10)
    train_parser.add_argument("--latent-dim", type=int, default=64)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--rotation-matrix-loss-weight", type=float, default=0.5)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument(
        "--virtual-frame-samples",
        type=int,
        default=64,
        help="Fibonacci virtual upper-shell frames per trajectory and epoch; 0 keeps the legacy object frame",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.set_defaults(function=train)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--output", default="")
    evaluate_parser.set_defaults(function=evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
