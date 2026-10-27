"""Train a point-cloud autoencoder for object-frame grasp geometry.

The input is ``object_pointcloud`` from the grasp dataset. Points remain in
the stored object coordinate frame. A PointNet-style encoder maps the unordered
point set to one latent vector, and an MLP decoder reconstructs an unordered
point set using symmetric Chamfer distance.

Examples
--------
Train an object point-cloud autoencoder::

    python -m memory.train_pointcloud_ae train \
        --dataset datasets/inspire_grasp.npz \
        --output-dir runs/object_pointcloud_ae \
        --latent-dim 128 --epochs 500

Export one latent vector per dataset sample::

    python -m memory.train_pointcloud_ae encode \
        --dataset datasets/inspire_grasp.npz \
        --checkpoint runs/object_pointcloud_ae/best.pt \
        --output runs/object_pointcloud_ae/inspire_latents.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


class PointCloudAutoEncoder(nn.Module):
    """Permutation-invariant point encoder and unordered point decoder."""

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        reconstruction_points: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.reconstruction_points = int(reconstruction_points)
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 128), nn.GELU(),
            nn.Linear(128, hidden_dim), nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, reconstruction_points * 3),
        )

    def encode(self, pointcloud: Tensor) -> Tensor:
        features = self.point_encoder(pointcloud)
        pooled = torch.cat((features.amax(dim=1), features.mean(dim=1)), dim=-1)
        return self.global_encoder(pooled)

    def decode(self, latent: Tensor) -> Tensor:
        points = self.decoder(latent)
        return points.reshape(-1, self.reconstruction_points, 3)

    def forward(self, pointcloud: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(pointcloud)
        return self.decode(latent), latent


def _build_model(config: dict[str, Any]) -> PointCloudAutoEncoder:
    return PointCloudAutoEncoder(
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        reconstruction_points=int(config["reconstruction_points"]),
    )


def _load_dataset(path: str) -> tuple[dict[str, np.ndarray], Tensor]:
    with np.load(path, allow_pickle=False) as loaded:
        if "object_pointcloud" not in loaded.files:
            raise ValueError("dataset must contain object_pointcloud")
        data = {key: loaded[key] for key in loaded.files}
    pointcloud = np.asarray(data["object_pointcloud"], dtype=np.float32)
    if pointcloud.ndim != 3 or pointcloud.shape[-1] != 3:
        raise ValueError(
            "object_pointcloud must have shape [samples, points, 3], "
            f"got {pointcloud.shape}"
        )
    if len(pointcloud) < 2:
        raise ValueError("at least two point clouds are required")
    if not np.isfinite(pointcloud).all():
        raise ValueError("object_pointcloud contains non-finite coordinates")
    frame = "object"
    sidecar = Path(path).with_suffix(".json")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text())
        frame = str(metadata.get("object_pointcloud_frame", frame))
    if frame != "object":
        raise ValueError(
            f"expected object-frame point clouds, metadata says {frame!r}"
        )
    return data, torch.from_numpy(pointcloud)


def _sample_points(pointcloud: Tensor, count: int) -> Tensor:
    batch, point_count, _ = pointcloud.shape
    if count <= point_count:
        indices = torch.rand(batch, point_count, device=pointcloud.device).argsort(dim=1)[:, :count]
        return pointcloud.gather(1, indices[..., None].expand(-1, -1, 3))
    indices = torch.randint(point_count, (batch, count), device=pointcloud.device)
    return pointcloud.gather(1, indices[..., None].expand(-1, -1, 3))


def _deterministic_points(pointcloud: Tensor, count: int) -> Tensor:
    if count >= pointcloud.shape[1]:
        return pointcloud
    indices = torch.linspace(
        0, pointcloud.shape[1] - 1, steps=count, device=pointcloud.device
    ).long()
    return pointcloud.index_select(1, indices)


def _chamfer_distance(prediction: Tensor, target: Tensor) -> Tensor:
    distances = torch.cdist(prediction, target, p=2).square()
    return distances.min(dim=2).values.mean() + distances.min(dim=1).values.mean()


def _split_indices(
    data: dict[str, np.ndarray], count: int, validation_fraction: float,
    seed: int, split_by_object: bool,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if split_by_object and "object_label_id" in data:
        groups = np.unique(data["object_label_id"])
        if len(groups) > 1:
            shuffled = torch.randperm(len(groups), generator=generator).numpy()
            valid_group_count = min(
                len(groups) - 1,
                max(1, int(math.ceil(len(groups) * validation_fraction))),
            )
            valid_groups = groups[shuffled[:valid_group_count]]
            valid_mask = np.isin(data["object_label_id"], valid_groups)
            valid = np.flatnonzero(valid_mask)
            train = np.flatnonzero(~valid_mask)
            return torch.from_numpy(train), torch.from_numpy(valid)
    permutation = torch.randperm(count, generator=generator)
    split = max(1, int(count * (1.0 - validation_fraction)))
    if split >= count:
        split = count - 1
    return permutation[:split], permutation[split:]


def _fit_epoch(
    model: PointCloudAutoEncoder,
    optimizer: torch.optim.Optimizer,
    pointclouds: Tensor,
    indices: Tensor,
    args: argparse.Namespace,
    device: torch.device,
    train: bool,
) -> float:
    model.train(train)
    loader = DataLoader(
        TensorDataset(indices), batch_size=args.batch_size,
        shuffle=train, drop_last=False,
    )
    losses: list[float] = []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for (sample_indices,) in loader:
            cloud = pointclouds[sample_indices].to(device)
            input_cloud = _sample_points(cloud, args.input_points)
            target_cloud = _sample_points(cloud, args.reconstruction_points)
            reconstruction, _ = model(input_cloud)
            loss = _chamfer_distance(reconstruction, target_cloud)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    data, pointclouds = _load_dataset(args.dataset)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if min(args.latent_dim, args.hidden_dim, args.input_points, args.reconstruction_points) < 1:
        raise ValueError("model dimensions and point counts must be positive")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs/batch-size must be positive and learning-rate must be positive")
    if args.input_points > pointclouds.shape[1] or args.reconstruction_points > pointclouds.shape[1]:
        raise ValueError(
            f"input/reconstruction point counts cannot exceed dataset point count {pointclouds.shape[1]}"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_indices, valid_indices = _split_indices(
        data, len(pointclouds), args.validation_fraction, args.seed,
        args.split_by_object,
    )
    device = torch.device(args.device)
    config = {
        "architecture": "pointnet_pointcloud_ae_v1",
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "input_points": args.input_points,
        "reconstruction_points": args.reconstruction_points,
        "coordinate_frame": "object",
        "pointcloud_key": "object_pointcloud",
        "loss": "symmetric_chamfer_squared",
        "split_by_object": args.split_by_object,
    }
    model = PointCloudAutoEncoder(
        args.latent_dim, args.hidden_dim, args.reconstruction_points
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name
    history_path = output_dir / "history.csv"
    best_validation = math.inf
    with history_path.open("w", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=["epoch", "train_chamfer", "validation_chamfer"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = _fit_epoch(
                model, optimizer, pointclouds, train_indices, args, device, True
            )
            validation_loss = _fit_epoch(
                model, optimizer, pointclouds, valid_indices, args, device, False
            )
            writer.writerow({
                "epoch": epoch, "train_chamfer": train_loss,
                "validation_chamfer": validation_loss,
            })
            history_file.flush()
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(
                    f"epoch {epoch:04d}/{args.epochs}: "
                    f"train_chamfer={train_loss:.6f} "
                    f"val_chamfer={validation_loss:.6f}"
                )
            if validation_loss < best_validation:
                best_validation = validation_loss
                torch.save({
                    "model": model.state_dict(), "config": config,
                    "epoch": epoch, "validation_chamfer": validation_loss,
                }, checkpoint_path)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved point-cloud AE checkpoint to {checkpoint_path}")


def encode(args: argparse.Namespace) -> None:
    data, pointclouds = _load_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    model = _build_model(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    latents = []
    with torch.no_grad():
        for start in range(0, len(pointclouds), args.batch_size):
            cloud = pointclouds[start:start + args.batch_size].to(args.device)
            cloud = _deterministic_points(cloud, int(config["input_points"]))
            latents.append(model.encode(cloud).cpu().numpy())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"latent": np.concatenate(latents).astype(np.float32)}
    for key in ("object_label", "object_label_id", "success"):
        if key in data:
            arrays[key] = data[key]
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps({
        "format": "DemoGrasp object point-cloud latents v1",
        "num_samples": len(pointclouds),
        "latent_dim": int(arrays["latent"].shape[-1]),
        "coordinate_frame": "object",
        "checkpoint": str(Path(args.checkpoint)),
    }, indent=2) + "\n")
    print(f"Saved point-cloud latents to {output}; shape={arrays['latent'].shape}")


def evaluate(args: argparse.Namespace) -> None:
    _, pointclouds = _load_dataset(args.dataset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    model = _build_model(config).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(pointclouds), args.batch_size):
            cloud = pointclouds[start:start + args.batch_size].to(args.device)
            input_cloud = _deterministic_points(cloud, int(config["input_points"]))
            target_cloud = _deterministic_points(
                cloud, int(config["reconstruction_points"])
            )
            reconstruction, _ = model(input_cloud)
            losses.append(float(_chamfer_distance(reconstruction, target_cloud).cpu()))
    result = {
        "samples": len(pointclouds),
        "pointcloud_shape": list(pointclouds.shape),
        "mean_chamfer": float(np.mean(losses)),
        "checkpoint_validation_chamfer": checkpoint.get("validation_chamfer"),
    }
    if args.sample_index < 0 or args.sample_index >= len(pointclouds):
        raise ValueError(
            f"sample-index must be in [0, {len(pointclouds) - 1}], got {args.sample_index}"
        )
    original = pointclouds[args.sample_index:args.sample_index + 1].to(args.device)
    input_cloud = _deterministic_points(original, int(config["input_points"]))
    target_cloud = _deterministic_points(
        original, int(config["reconstruction_points"])
    )
    with torch.no_grad():
        reconstruction, _ = model(input_cloud)
    original_np = original[0].cpu().numpy()
    reconstruction_np = reconstruction[0].cpu().numpy()
    result["visualized_sample_index"] = int(args.sample_index)
    result["visualized_sample_chamfer"] = float(
        _chamfer_distance(reconstruction, target_cloud).cpu()
    )

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 5), constrained_layout=True)
    limits = np.concatenate((original_np, reconstruction_np), axis=0)
    lower = limits.min(axis=0)
    upper = limits.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float((upper - lower).max()) * 0.55, 1e-4)
    for panel, points, title, color in (
        (1, original_np, "Original object-frame point cloud", "#2563eb"),
        (2, reconstruction_np, "Autoencoder reconstruction", "#f97316"),
    ):
        axis = figure.add_subplot(1, 2, panel, projection="3d")
        axis.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.8, c=color)
        axis.set_title(title)
        axis.set_xlabel("X (m)")
        axis.set_ylabel("Y (m)")
        axis.set_zlabel("Z (m)")
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
    figure.suptitle(
        f"Sample {args.sample_index} | Chamfer={result['visualized_sample_chamfer']:.6g}"
    )
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("Showing interactive point-cloud comparison; close the window to finish.")
    plt.show(block=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset", required=True)
        subparser.add_argument("--batch-size", type=int, default=64)
        subparser.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--output-dir", default="runs/object_pointcloud_ae")
    train_parser.add_argument("--checkpoint-name", default="best.pt")
    train_parser.add_argument("--epochs", type=int, default=500)
    train_parser.add_argument("--validation-fraction", type=float, default=0.1)
    train_parser.add_argument("--latent-dim", type=int, default=128)
    train_parser.add_argument("--hidden-dim", type=int, default=256)
    train_parser.add_argument("--input-points", type=int, default=256)
    train_parser.add_argument("--reconstruction-points", type=int, default=256)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-6)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--log-every", type=int, default=10)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--split-by-object", dest="split_by_object", action="store_true",
        help="hold out complete object identities for validation (default)",
    )
    train_parser.add_argument(
        "--random-sample-split", dest="split_by_object", action="store_false",
        help="split samples randomly instead of holding out object identities",
    )
    train_parser.set_defaults(split_by_object=True)
    train_parser.set_defaults(function=train)

    for mode, function in (("encode", encode), ("evaluate", evaluate)):
        mode_parser = subparsers.add_parser(mode)
        add_common(mode_parser)
        mode_parser.add_argument("--checkpoint", required=True)
        if mode == "encode":
            mode_parser.add_argument("--output", required=True)
        else:
            mode_parser.add_argument("--output", default="")
            mode_parser.add_argument("--sample-index", type=int, default=0)
        mode_parser.set_defaults(function=function)

    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
