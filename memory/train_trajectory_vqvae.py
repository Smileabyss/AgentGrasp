"""Learn discrete tokens for DemoGrasp trajectories.

The input trajectory is, in order:

    wrist position in the initial object frame (3)
    wrist rotation in the initial object frame as 6D rotation (6)
    raw active hand joint positions (6 for the Inspire hand)

No data-dependent normalization is applied.  A temporal VQ-VAE compresses a
50-step trajectory into ``num_tokens`` discrete codebook indices.

Examples
--------
Train a tokenizer::

    python -m memory.train_trajectory_vqvae train \
        --dataset datasets/inspire_grasp.npz \
        --output-dir runs/trajectory_vqvae --epochs 200

Export discrete tokens after training::

    python -m memory.train_trajectory_vqvae encode \
        --dataset datasets/inspire_grasp.npz \
        --checkpoint runs/trajectory_vqvae/best.pt \
        --output datasets/inspire_grasp_tokens.npz
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def quaternion_conjugate(q: Tensor) -> Tensor:
    out = q.clone()
    out[..., :3] *= -1
    return out


def quaternion_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Convert Isaac Gym's xyzw quaternion convention to a rotation matrix."""

    q = quaternion_normalize(q)
    x, y, z, w = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quaternion_to_rotation6d(q: Tensor) -> Tensor:
    matrix = quaternion_to_matrix(q)
    # First two columns, flattened in column-major order.
    return matrix[..., :, :2].transpose(-1, -2).reshape(q.shape[:-1] + (6,))


def rotation6d_to_matrix(rotation6d: Tensor) -> Tensor:
    """Recover a proper rotation matrix with Gram-Schmidt orthogonalization."""

    first = torch.nn.functional.normalize(rotation6d[..., :3], dim=-1)
    second_raw = rotation6d[..., 3:]
    second = torch.nn.functional.normalize(
        second_raw - first * (first * second_raw).sum(dim=-1, keepdim=True), dim=-1
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def prepare_trajectories(data: Dict[str, np.ndarray]) -> Tensor:
    """Build [N, T, 9 + hand_joint_count] trajectories."""

    wrist = torch.from_numpy(np.asarray(data["wrist_trajectory"], dtype=np.float32))
    object_pose = torch.from_numpy(np.asarray(data["object_pose_world"], dtype=np.float32))
    joints = torch.from_numpy(
        np.asarray(data["palm_joint_position_trajectory"], dtype=np.float32)
    )
    if wrist.ndim != 3 or wrist.shape[-1] != 7:
        raise ValueError("wrist_trajectory must have shape [N, T, 7] in xyz+xyzw format")
    if object_pose.ndim != 2 or object_pose.shape[-1] != 7:
        raise ValueError("object_pose_world must have shape [N, 7] in xyz+xyzw format")
    if joints.shape[0] != wrist.shape[0] or joints.ndim != 3:
        raise ValueError("palm_joint_position_trajectory must have shape [N, T, D]")
    if joints.shape[1] != wrist.shape[1]:
        # Older collector outputs may contain full-length hand joints alongside
        # already-trimmed/resampled wrist poses. Resample joints to the wrist
        # sequence length so the policy input and target have one time axis.
        source_time = torch.linspace(0.0, 1.0, joints.shape[1], dtype=joints.dtype)
        target_time = torch.linspace(0.0, 1.0, wrist.shape[1], dtype=joints.dtype)
        right = torch.searchsorted(source_time, target_time).clamp(1, joints.shape[1] - 1)
        left = right - 1
        weight = ((target_time - source_time[left]) / (source_time[right] - source_time[left]))
        joints = joints[:, left] * (1.0 - weight[None, :, None]) + joints[:, right] * weight[None, :, None]

    object_position = object_pose[:, None, :3]
    object_quaternion = quaternion_normalize(object_pose[:, 3:7])
    wrist_position = wrist[..., :3]
    wrist_quaternion = quaternion_normalize(wrist[..., 3:7])

    # World -> initial object frame: R_obj^T (p_world - p_obj).
    object_rotation = quaternion_to_matrix(object_quaternion)
    position_object = torch.matmul(
        (wrist_position - object_position).unsqueeze(-2), object_rotation[:, None]
    ).squeeze(-2)
    relative_quaternion = quaternion_multiply(
        quaternion_conjugate(object_quaternion)[:, None], wrist_quaternion
    )
    rotation6d_object = quaternion_to_rotation6d(relative_quaternion)
    return torch.cat((position_object, rotation6d_object, joints), dim=-1)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, latent_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.commitment_cost = commitment_cost
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # z: [B, K, C], quantize every temporal latent independently.
        distances = (
            z.square().sum(-1, keepdim=True)
            - 2 * z @ self.embedding.weight.t()
            + self.embedding.weight.square().sum(-1)
        )
        indices = distances.argmin(dim=-1)
        quantized = self.embedding(indices)
        codebook_loss = (quantized - z.detach()).square().mean()
        commitment_loss = self.commitment_cost * (z - quantized.detach()).square().mean()
        quantized_st = z + (quantized - z).detach()
        return quantized_st, codebook_loss + commitment_loss, indices, quantized


class TemporalVQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        num_tokens: int = 10,
        latent_dim: int = 64,
        codebook_size: int = 128,
        hidden_dim: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, latent_dim, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(num_tokens),
        )
        self.quantizer = VectorQuantizer(codebook_size, latent_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, input_dim, 5, padding=2),
        )

    def encode(self, trajectory: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        z = self.encoder(trajectory.transpose(1, 2)).transpose(1, 2)
        quantized, vq_loss, indices, _ = self.quantizer(z)
        return quantized, vq_loss, indices

    def decode(self, quantized: Tensor) -> Tensor:
        upsampled = torch.nn.functional.interpolate(
            quantized.transpose(1, 2),
            size=self.sequence_length,
            mode="linear",
            align_corners=False,
        )
        return self.decoder(upsampled).transpose(1, 2)

    def forward(self, trajectory: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        quantized, vq_loss, indices = self.encode(trajectory)
        return self.decode(quantized), vq_loss, indices


def _load_data(path: str) -> Tuple[Dict[str, np.ndarray], Tensor]:
    with np.load(path, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    trajectories = prepare_trajectories(data)
    return data, trajectories


def _build_model(config: Dict) -> TemporalVQVAE:
    return TemporalVQVAE(
        input_dim=int(config["input_dim"]),
        sequence_length=int(config["sequence_length"]),
        num_tokens=int(config["num_tokens"]),
        latent_dim=int(config["latent_dim"]),
        codebook_size=int(config["codebook_size"]),
        hidden_dim=int(config["hidden_dim"]),
        commitment_cost=float(config["commitment_cost"]),
    )


def train(args: argparse.Namespace) -> None:
    data, trajectories = _load_data(args.dataset)
    if trajectories.shape[1] != args.sequence_length:
        raise ValueError(
            f"Expected {args.sequence_length} trajectory steps, got {trajectories.shape[1]}"
        )
    if trajectories.shape[0] < 2:
        raise ValueError("At least two trajectories are required for a train/validation split")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    permutation = torch.randperm(trajectories.shape[0])
    split = max(1, int(trajectories.shape[0] * (1.0 - args.validation_fraction)))
    train_data = trajectories[permutation[:split]]
    valid_data = trajectories[permutation[split:]]
    if valid_data.shape[0] == 0:
        valid_data = train_data[:1]

    model_config = {
        "input_dim": trajectories.shape[-1],
        "sequence_length": trajectories.shape[1],
        "num_tokens": args.num_tokens,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "hidden_dim": args.hidden_dim,
        "commitment_cost": args.commitment_cost,
        "vq_loss_weight": args.vq_loss_weight,
        "feature_order": ["wrist_position_object", "wrist_rotation6d_object", "raw_hand_joints"],
        "normalization": "none; wrist position is relative to the initial object frame",
    }
    model = _build_model(model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        TensorDataset(train_data), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation = math.inf
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for (batch,) in loader:
            batch = batch.to(args.device)
            reconstruction, vq_loss, _ = model(batch)
            reconstruction_loss = torch.mean((reconstruction - batch).square())
            loss = reconstruction_loss + args.vq_loss_weight * vq_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            # Keep codebook vectors in a stable numerical range.
            with torch.no_grad():
                model.quantizer.embedding.weight.clamp_(-1.0, 1.0)
            train_total += loss.item() * batch.shape[0]

        model.eval()
        with torch.no_grad():
            validation = []
            for start in range(0, valid_data.shape[0], args.batch_size):
                batch = valid_data[start : start + args.batch_size].to(args.device)
                reconstruction, vq_loss, _ = model(batch)
                validation.append(
                    (
                        torch.mean((reconstruction - batch).square())
                        + args.vq_loss_weight * vq_loss
                    ).item()
                )
            validation_loss = float(np.mean(validation))
        row = {
            "epoch": float(epoch),
            "train_loss": train_total / train_data.shape[0],
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            torch.save(
                {"model": model.state_dict(), "config": model_config, "history": history},
                output_dir / "best.pt",
            )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d}/{args.epochs}: "
                f"train={row['train_loss']:.6f} val={validation_loss:.6f}"
            )

    (output_dir / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    print(f"Saved tokenizer checkpoint to {output_dir / 'best.pt'}")


def encode(args: argparse.Namespace) -> None:
    data, trajectories = _load_data(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = _build_model(checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = []
    with torch.no_grad():
        for start in range(0, trajectories.shape[0], args.batch_size):
            batch = trajectories[start : start + args.batch_size].to(args.device)
            _, _, indices = model(batch)
            tokens.append(indices.cpu().numpy().astype(np.int64))
    tokens_array = np.concatenate(tokens, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        object_label=data["object_label"],
        object_label_id=data["object_label_id"],
        tokens=tokens_array,
        success=data["success"],
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "format": "DemoGrasp discrete trajectory tokens v1",
                "num_samples": int(tokens_array.shape[0]),
                "num_tokens": int(tokens_array.shape[1]),
                "codebook_size": int(checkpoint["config"]["codebook_size"]),
                "feature_order": checkpoint["config"]["feature_order"],
                "normalization": checkpoint["config"]["normalization"],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved discrete trajectory tokens: {output} shape={tokens_array.shape}")


def evaluate(args: argparse.Namespace) -> None:
    """Report reconstruction quality and codebook utilization."""

    data, trajectories = _load_data(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = _build_model(checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    reconstructions = []
    token_batches = []
    vq_losses = []
    with torch.no_grad():
        for start in range(0, trajectories.shape[0], args.batch_size):
            batch = trajectories[start : start + args.batch_size].to(args.device)
            reconstruction, vq_loss, indices = model(batch)
            reconstructions.append(reconstruction.cpu())
            token_batches.append(indices.cpu())
            vq_losses.append(float(vq_loss))
    reconstruction = torch.cat(reconstructions, dim=0)
    tokens = torch.cat(token_batches, dim=0)
    error = reconstruction - trajectories

    groups = {
        "wrist_position_object": slice(0, 3),
        "wrist_rotation6d_object": slice(3, 9),
        "raw_hand_joints": slice(9, trajectories.shape[-1]),
    }
    metrics = {
        "samples": int(trajectories.shape[0]),
        "trajectory_shape": list(trajectories.shape),
        "reconstruction_mse": float(error.square().mean()),
        "reconstruction_rmse": float(error.square().mean().sqrt()),
        "mean_vq_loss": float(np.mean(vq_losses)),
        "used_codebook_entries": int(torch.unique(tokens).numel()),
        "codebook_size": int(model.codebook_size),
    }
    probabilities = torch.bincount(
        tokens.reshape(-1), minlength=model.codebook_size
    ).float()
    probabilities /= probabilities.sum().clamp_min(1.0)
    metrics["codebook_perplexity"] = float(
        torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    )
    for name, feature_slice in groups.items():
        group_error = error[..., feature_slice]
        metrics[f"{name}_mse"] = float(group_error.square().mean())
        metrics[f"{name}_rmse"] = float(group_error.square().mean().sqrt())
    true_rotation = rotation6d_to_matrix(trajectories[..., 3:9])
    predicted_rotation = rotation6d_to_matrix(reconstruction[..., 3:9])
    relative_rotation = torch.matmul(true_rotation.transpose(-1, -2), predicted_rotation)
    trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    rotation_angle = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    metrics["wrist_rotation_geodesic_mean_deg"] = float(rotation_angle.mean() * 180.0 / np.pi)
    metrics["wrist_rotation_geodesic_median_deg"] = float(
        rotation_angle.median() * 180.0 / np.pi
    )
    metrics["wrist_rotation_geodesic_95th_deg"] = float(
        torch.quantile(rotation_angle, 0.95) * 180.0 / np.pi
    )
    metrics["wrist_position_final_rmse"] = float(error[:, -1, :3].square().mean().sqrt())
    metrics["raw_hand_joints_final_rmse"] = float(error[:, -1, 9:].square().mean().sqrt())

    print(json.dumps(metrics, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"Saved evaluation metrics to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--batch-size", type=int, default=128)
        subparser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--output-dir", default="runs/trajectory_vqvae")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--sequence-length", type=int, default=20)
    train_parser.add_argument("--num-tokens", type=int, default=10)
    train_parser.add_argument("--latent-dim", type=int, default=64)
    train_parser.add_argument("--codebook-size", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--commitment-cost", type=float, default=0.25)
    train_parser.add_argument(
        "--vq-loss-weight",
        type=float,
        default=0.1,
        help="Weight of the codebook and commitment losses (default: 0.1).",
    )
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.set_defaults(function=train)

    encode_parser = subparsers.add_parser("encode")
    add_common(encode_parser)
    encode_parser.add_argument("--checkpoint", required=True)
    encode_parser.add_argument("--output", required=True)
    encode_parser.set_defaults(function=encode)

    evaluate_parser = subparsers.add_parser("evaluate")
    add_common(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--output", default="")
    evaluate_parser.set_defaults(function=evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
