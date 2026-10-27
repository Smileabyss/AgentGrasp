"""Fit a small RealNVP prior to the latent codes of a trained trajectory AE.

The AE is frozen.  Training maps flattened AE latents ``z`` to a standard
Gaussian ``u`` with an invertible affine coupling flow.  Sampling reverses the
map and decodes the resulting latent with the original AE decoder.

Examples
--------
python -m memory.train_latent_flow train \
  --dataset datasets/inspire_grasp.npz \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --output-dir runs/latent_flow --epochs 500

python -m memory.train_latent_flow sample \
  --dataset datasets/inspire_grasp.npz --object-index 0 \
  --ae-checkpoint runs/trajectory_ae/best.pt \
  --flow-checkpoint runs/latent_flow/best.pt \
  --num-samples 32 --output runs/latent_flow/samples.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from memory.train_trajectory_ae import (
    _build_model,
    _load_data,
    _load_world_data,
    _sample_virtual_frames,
    _sliding_trajectory_windows,
)
from memory.train_pointcloud_ae import (
    PointCloudAutoEncoder,
    _build_model as _build_pointcloud_model,
    _deterministic_points,
    _load_dataset as _load_pointcloud_dataset,
)


class AffineCoupling(nn.Module):
    def __init__(
        self, dim: int, mask: Tensor, hidden_dim: int, scale_limit: float,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.register_buffer("mask", mask.reshape(1, dim))
        self.scale_limit = float(scale_limit)
        self.condition_dim = int(condition_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + self.condition_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _conditioned_input(self, masked: Tensor, condition: Tensor | None) -> Tensor:
        if self.condition_dim == 0:
            return masked
        if condition is None:
            raise ValueError("this flow requires point-cloud condition features")
        if condition.ndim != 2 or condition.shape != (masked.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must have shape ({masked.shape[0]}, {self.condition_dim}), "
                f"got {tuple(condition.shape)}"
            )
        return torch.cat((masked, condition), dim=-1)

    def forward(
        self, x: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        masked = x * self.mask
        shift, scale = self.net(self._conditioned_input(masked, condition)).chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return masked + (1.0 - self.mask) * (x * torch.exp(scale) + shift), scale.sum(-1)

    def inverse(self, y: Tensor, condition: Tensor | None = None) -> Tensor:
        masked = y * self.mask
        shift, scale = self.net(self._conditioned_input(masked, condition)).chunk(2, dim=-1)
        scale = self.scale_limit * torch.tanh(scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-scale))


class RealNVP(nn.Module):
    def __init__(
        self, dim: int, layers: int = 6, hidden_dim: int = 128,
        scale_limit: float = 2.0, condition_dim: int = 0,
    ):
        super().__init__()
        if dim < 2 or layers < 1:
            raise ValueError("dim must be >= 2 and layers must be positive")
        masks = []
        for index in range(layers):
            mask = torch.zeros(dim)
            mask[index % 2 :: 2] = 1.0
            masks.append(mask)
        self.layers = nn.ModuleList(
            [
                AffineCoupling(
                    dim, mask, hidden_dim, scale_limit, condition_dim
                )
                for mask in masks
            ]
        )
        self.dim = dim
        self.condition_dim = int(condition_dim)

    def forward(
        self, x: Tensor, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, contribution = layer(x, condition)
            log_det = log_det + contribution
        return x, log_det

    def inverse(self, y: Tensor, condition: Tensor | None = None) -> Tensor:
        for layer in reversed(self.layers):
            y = layer.inverse(y, condition)
        return y


def _standard_normal_log_prob(x: Tensor) -> Tensor:
    return -0.5 * (x.square() + math.log(2.0 * math.pi)).sum(-1)


def _load_latents(
    args: argparse.Namespace, device: torch.device
) -> tuple[Tensor, Tensor, dict, Tensor, Tensor]:
    checkpoint = torch.load(args.ae_checkpoint, map_location=device)
    traj_length = int(checkpoint["config"]["sequence_length"])
    if args.traj_length is not None and args.traj_length != traj_length:
        raise ValueError(
            f"--traj_length={args.traj_length} does not match the AE checkpoint "
            f"sequence length ({traj_length})"
        )
    with np.load(args.dataset, allow_pickle=False) as loaded:
        if "object_label_id" in loaded.files:
            source_object_ids = torch.from_numpy(
                np.asarray(loaded["object_label_id"], dtype=np.int64)
            )
        elif "object_label" in loaded.files:
            _, source_object_ids = np.unique(loaded["object_label"], return_inverse=True)
            source_object_ids = torch.from_numpy(source_object_ids.astype(np.int64))
        else:
            source_object_ids = None
    if checkpoint["config"].get("coordinate_frame") == "virtual_object_pointcloud":
        wrist, object_pose, joints, pointcloud = _load_world_data(args.dataset)
        samples = int(args.virtual_frame_samples or checkpoint["config"].get("virtual_frame_samples", 1))
        window_count = wrist.shape[1] - traj_length + 1
        if window_count < 1:
            raise ValueError(
                f"AE checkpoint trajectory length {traj_length} exceeds dataset length {wrist.shape[1]}"
            )
        source_count = wrist.shape[0]
        dataset_source_count = source_count
        source_ids = torch.arange(source_count)
        if source_object_ids is None:
            source_object_ids = torch.arange(source_count)
        elif len(source_object_ids) != source_count:
            raise ValueError("object identity count does not match trajectory count")
        if args.max_latent_samples > 0:
            selected_source_count = min(
                source_count,
                max(1, int(math.ceil(args.max_latent_samples / (samples * window_count)))),
            )
            source_indices = torch.randperm(source_count)[:selected_source_count]
            source_ids = source_ids[source_indices]
            wrist, object_pose, joints, pointcloud = (
                value[source_indices] for value in (wrist, object_pose, joints, pointcloud)
            )
            source_count = selected_source_count
        trajectories = _sample_virtual_frames(
            wrist.to(device), object_pose.to(device), joints.to(device),
            pointcloud.to(device), samples,
        ).cpu()
        group_ids = source_object_ids[source_ids].repeat_interleave(samples * window_count)
        condition_source_ids = source_ids.repeat_interleave(samples * window_count)
    else:
        trajectories = _load_data(args.dataset)
        if trajectories.shape[1] < traj_length:
            raise ValueError(
                f"AE checkpoint trajectory length {traj_length} exceeds dataset length {trajectories.shape[1]}"
            )
        window_count = trajectories.shape[1] - traj_length + 1
        dataset_source_count = trajectories.shape[0]
        if source_object_ids is None:
            source_object_ids = torch.arange(trajectories.shape[0])
        elif len(source_object_ids) != trajectories.shape[0]:
            raise ValueError("object identity count does not match trajectory count")
        group_ids = source_object_ids.repeat_interleave(window_count)
        condition_source_ids = torch.arange(trajectories.shape[0]).repeat_interleave(window_count)
    trajectories = _sliding_trajectory_windows(trajectories, traj_length)
    ae = _build_model(checkpoint["config"]).to(device)
    ae.load_state_dict(checkpoint["model"])
    ae.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, trajectories.shape[0], args.batch_size):
            batches.append(ae.encode(trajectories[start : start + args.batch_size].to(device)).cpu())
    latent = torch.cat(batches, dim=0)
    if args.max_latent_samples > 0 and latent.shape[0] > args.max_latent_samples:
        keep = torch.randperm(latent.shape[0])[: args.max_latent_samples]
        latent = latent[keep]
        group_ids = group_ids[keep]
        condition_source_ids = condition_source_ids[keep]
    _, group_ids = torch.unique(group_ids, sorted=True, return_inverse=True)
    shape = tuple(latent.shape[1:])
    return latent.reshape(latent.shape[0], -1), trajectories, {
        "latent_shape": shape,
        "coordinate_frame": checkpoint["config"].get("coordinate_frame", "initial_object"),
        "virtual_frame_samples": samples if checkpoint["config"].get("coordinate_frame") == "virtual_object_pointcloud" else 0,
        "sequence_length": traj_length,
        "window_stride": 1,
        "source_sample_count": dataset_source_count,
    }, group_ids, condition_source_ids


def _encode_pointcloud_conditions(
    dataset: str,
    checkpoint_path: str,
    sample_ids: Tensor,
    expected_source_count: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, dict]:
    point_data, pointclouds = _load_pointcloud_dataset(dataset)
    if len(pointclouds) != expected_source_count:
        raise ValueError(
            "trajectory and object_pointcloud sample counts differ: "
            f"{expected_source_count} vs {len(pointclouds)}"
        )
    if sample_ids.ndim != 1 or sample_ids.numel() == 0:
        raise ValueError("point-cloud condition indices must be a non-empty vector")
    if int(sample_ids.min()) < 0 or int(sample_ids.max()) >= len(pointclouds):
        raise ValueError(
            "trajectory samples do not align with the dataset's object_pointcloud rows"
        )
    for key in ("object_label", "object_label_id"):
        if key in point_data and len(point_data[key]) != len(pointclouds):
            raise ValueError(f"{key} count does not match object_pointcloud count")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    point_config = checkpoint["config"]
    if point_config.get("coordinate_frame") != "object":
        raise ValueError("point-cloud AE must be trained in the object coordinate frame")
    model: PointCloudAutoEncoder = _build_pointcloud_model(point_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    unique_ids, inverse_ids = torch.unique(sample_ids, sorted=True, return_inverse=True)
    selected = pointclouds[unique_ids]
    conditions = []
    with torch.no_grad():
        for start in range(0, len(selected), batch_size):
            cloud = selected[start:start + batch_size].to(device)
            cloud = _deterministic_points(cloud, int(point_config["input_points"]))
            conditions.append(model.encode(cloud).cpu())
    encoded_unique = torch.cat(conditions, dim=0)
    return encoded_unique[inverse_ids], {
        "condition_dim": int(point_config["latent_dim"]),
        "pointcloud_coordinate_frame": "object",
        "pointcloud_ae_checkpoint": str(Path(checkpoint_path)),
        "pointcloud_ae_architecture": point_config.get("architecture"),
    }


def _make_flow(config: dict, device: torch.device) -> RealNVP:
    return RealNVP(
        int(config["latent_dim"]), int(config["layers"]),
        int(config["hidden_dim"]), float(config["scale_limit"]),
        int(config.get("condition_dim", 0)),
    ).to(device)


def train(args: argparse.Namespace) -> None:
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.max_grad_norm <= 0:
        raise ValueError("--learning-rate and --max-grad-norm must be positive")
    device = torch.device(args.device)
    latents, _, info, group_ids, condition_source_ids = _load_latents(args, device)
    condition_info = None
    if args.pointcloud_ae_checkpoint:
        conditions, condition_info = _encode_pointcloud_conditions(
            args.dataset, args.pointcloud_ae_checkpoint, condition_source_ids,
            info["source_sample_count"], args.batch_size, device,
        )
    else:
        conditions = torch.empty((len(latents), 0), dtype=latents.dtype)
    if len(conditions) != len(latents):
        raise RuntimeError("trajectory latent and point-cloud condition counts differ")
    source_count = int(group_ids.max()) + 1
    permutation = torch.randperm(source_count)
    split = max(1, int(source_count * (1.0 - args.validation_fraction)))
    train_group_mask = torch.zeros(source_count, dtype=torch.bool)
    train_group_mask[permutation[:split]] = True
    train_rows = train_group_mask[group_ids]
    train_data, valid_data = latents[train_rows], latents[~train_rows]
    train_conditions, valid_conditions = conditions[train_rows], conditions[~train_rows]
    if valid_data.shape[0] == 0:
        valid_data, valid_conditions = train_data[:1], train_conditions[:1]
    config = {
        "latent_dim": int(latents.shape[1]), "latent_shape": list(info["latent_shape"]),
        "layers": args.layers, "hidden_dim": args.hidden_dim, "scale_limit": args.scale_limit,
        "base_distribution": "standard_normal",
        "conditional": condition_info is not None,
        "condition_dim": 0 if condition_info is None else int(condition_info["condition_dim"]),
        "condition_type": (
            None if condition_info is None else "frozen_object_pointcloud_ae_latent"
        ),
        "pointcloud_coordinate_frame": (
            None if condition_info is None else condition_info["pointcloud_coordinate_frame"]
        ),
        "pointcloud_ae_checkpoint": (
            None if condition_info is None else condition_info["pointcloud_ae_checkpoint"]
        ),
        "pointcloud_ae_architecture": (
            None if condition_info is None else condition_info["pointcloud_ae_architecture"]
        ),
        "coordinate_frame": info["coordinate_frame"],
        "virtual_frame_samples": info["virtual_frame_samples"],
        "sequence_length": info["sequence_length"],
        "window_stride": info["window_stride"],
    }
    flow = _make_flow(config, device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(train_data, train_conditions),
        batch_size=args.batch_size, shuffle=True,
    )
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name
    best = float("inf"); history = []
    for epoch in range(1, args.epochs + 1):
        flow.train(); train_losses = []
        for batch, condition in loader:
            u, log_det = flow(batch.to(device), condition.to(device))
            loss = -(_standard_normal_log_prob(u) + log_det).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), args.max_grad_norm); optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        flow.eval()
        with torch.no_grad():
            u, log_det = flow(valid_data.to(device), valid_conditions.to(device))
            validation = float(-(_standard_normal_log_prob(u) + log_det).mean().cpu())
        row = {"epoch": epoch, "train_nll": float(np.mean(train_losses)), "validation_nll": validation}
        history.append(row)
        if validation < best:
            best = validation
            torch.save({"model": flow.state_dict(), "config": config, "history": history}, checkpoint_path)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d}/{args.epochs}: train_nll={row['train_nll']:.4f} val_nll={validation:.4f}")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved latent flow checkpoint to {checkpoint_path}")


def sample(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint = torch.load(args.flow_checkpoint, map_location=device)
    flow = _make_flow(checkpoint["config"], device); flow.load_state_dict(checkpoint["model"]); flow.eval()
    ae_checkpoint = torch.load(args.ae_checkpoint, map_location=device)
    ae = _build_model(ae_checkpoint["config"]).to(device); ae.load_state_dict(ae_checkpoint["model"]); ae.eval()
    shape = tuple(checkpoint["config"]["latent_shape"])
    expected_shape = (
        int(ae_checkpoint["config"]["num_tokens"]),
        int(ae_checkpoint["config"]["latent_dim"]),
    )
    if shape != expected_shape:
        raise ValueError(
            f"flow latent shape {shape} does not match trajectory AE {expected_shape}"
        )
    condition = None
    object_label = ""
    object_label_id = -1
    if flow.condition_dim > 0:
        if not args.dataset:
            raise ValueError("conditional flow sampling requires --dataset")
        pointcloud_checkpoint_path = (
            args.pointcloud_ae_checkpoint
            or checkpoint["config"].get("pointcloud_ae_checkpoint")
        )
        if not pointcloud_checkpoint_path:
            raise ValueError(
                "conditional flow sampling requires --pointcloud-ae-checkpoint"
            )
        point_data, pointclouds = _load_pointcloud_dataset(args.dataset)
        if args.object_index < 0 or args.object_index >= len(pointclouds):
            raise ValueError(
                f"object-index must be in [0, {len(pointclouds) - 1}], got {args.object_index}"
            )
        point_checkpoint = torch.load(pointcloud_checkpoint_path, map_location=device)
        point_config = point_checkpoint["config"]
        if point_config.get("coordinate_frame") != "object":
            raise ValueError("point-cloud AE must be trained in the object coordinate frame")
        point_encoder = _build_pointcloud_model(point_config).to(device)
        point_encoder.load_state_dict(point_checkpoint["model"])
        point_encoder.eval()
        cloud = pointclouds[args.object_index:args.object_index + 1].to(device)
        cloud = _deterministic_points(cloud, int(point_config["input_points"]))
        with torch.no_grad():
            condition = point_encoder.encode(cloud)
        if condition.shape[-1] != flow.condition_dim:
            raise ValueError(
                f"point-cloud latent dim {condition.shape[-1]} does not match flow condition dim {flow.condition_dim}"
            )
        if "object_label" in point_data:
            object_label = str(point_data["object_label"][args.object_index])
        if "object_label_id" in point_data:
            object_label_id = int(point_data["object_label_id"][args.object_index])
    with torch.no_grad():
        base = torch.randn(args.num_samples, flow.dim, device=device)
        expanded_condition = (
            None if condition is None
            else condition.expand(args.num_samples, -1)
        )
        latent = flow.inverse(base, expanded_condition).reshape(args.num_samples, *shape)
        trajectories = ae.decode(latent)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "latent": latent.cpu().numpy(),
        "trajectories": trajectories.cpu().numpy(),
        "object_index": np.asarray(args.object_index, dtype=np.int64),
        "object_label": np.asarray(object_label),
        "object_label_id": np.asarray(object_label_id, dtype=np.int64),
    }
    if condition is not None:
        result["condition"] = condition.cpu().numpy()
    np.savez_compressed(output, **result)
    print(f"Saved {args.num_samples} decoded trajectories to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    def common(p):
        p.add_argument("--ae-checkpoint", required=True); p.add_argument("--batch-size", type=int, default=512)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser = sub.add_parser("train"); common(train_parser)
    train_parser.add_argument("--dataset", required=True); train_parser.add_argument("--output-dir", default="runs/latent_flow")
    train_parser.add_argument(
        "--pointcloud-ae-checkpoint", default="",
        help="optional frozen object-frame point-cloud AE; omit for an unconditional Flow",
    )
    train_parser.add_argument("--checkpoint-name", default="best.pt")
    train_parser.add_argument("--epochs", type=int, default=500); train_parser.add_argument("--validation-fraction", type=float, default=.1)
    train_parser.add_argument(
        "--traj_length", "--traj-length", dest="traj_length", type=int, default=None,
        help="trajectory window length; defaults to the AE checkpoint sequence length",
    )
    train_parser.add_argument("--layers", type=int, default=6); train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--scale-limit", type=float, default=2.0); train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-6); train_parser.add_argument("--max-grad-norm", type=float, default=5.0)
    train_parser.add_argument(
        "--virtual-frame-samples", type=int, default=None,
        help="override the virtual-frame views per trajectory when the AE uses virtual frames",
    )
    train_parser.add_argument("--log-every", type=int, default=25); train_parser.set_defaults(function=train)
    train_parser.add_argument(
        "--max-latent-samples", type=int, default=0,
        help="optional cap after window expansion; 0 keeps every trajectory window/view",
    )
    sample_parser = sub.add_parser("sample"); common(sample_parser); sample_parser.add_argument("--flow-checkpoint", required=True)
    sample_parser.add_argument("--dataset", default="",
                               help="required for a conditional flow; supplies object_pointcloud")
    sample_parser.add_argument("--pointcloud-ae-checkpoint", default="",
                               help="defaults to the checkpoint path recorded by training")
    sample_parser.add_argument("--object-index", type=int, default=0)
    sample_parser.add_argument("--num-samples", type=int, default=32); sample_parser.add_argument("--output", default="runs/latent_flow/samples.npz"); sample_parser.set_defaults(function=sample)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
