"""Jointly train PoinTr completion and point-cloud canonicalization stages."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from perception.rgbd_frame import prepare_two_stage_clouds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POINTR_ROOT = PROJECT_ROOT / "external" / "PoinTr"
POINTR_CONFIG = POINTR_ROOT / "cfgs" / "PCN_models" / "PoinTr.yaml"
DEFAULT_PRETRAINED = POINTR_ROOT / "pretrained" / "PCN_Pretrained.pth"


class RGBDPointCloudDataset(Dataset):
    """Prepare partial input and full targets in estimated and object frames."""

    def __init__(self, shard_paths, intrinsics, num_points=512, max_depth=5.0, seed=42):
        self.shard_paths = list(shard_paths)
        self.intrinsics = np.asarray(intrinsics, dtype=np.float32)
        self.num_points = num_points
        self.max_depth = max_depth
        self.counts = []
        for path in self.shard_paths:
            with np.load(path) as shard:
                required = {"depth", "object_mask", "points_camera", "pose_camera"}
                missing = required.difference(shard.files)
                if missing:
                    raise ValueError("{} is missing required fields: {}".format(path, sorted(missing)))
                if shard["points_camera"].ndim != 3 or shard["points_camera"].shape[-1] != 3:
                    raise ValueError("{} points_camera must have shape [samples, points, 3]".format(path))
                self.counts.append(len(shard["points_camera"]))

        self.offsets = np.cumsum([0] + self.counts).tolist()
        total = self.offsets[-1]
        self.partial_clouds = np.empty((total, num_points, 3), dtype=np.float32)
        self.complete_estimated = np.empty((total, num_points, 3), dtype=np.float32)
        self.complete_object = np.empty((total, num_points, 3), dtype=np.float32)
        rng = np.random.default_rng(seed)

        for shard_id, path in enumerate(self.shard_paths):
            start = self.offsets[shard_id]
            with np.load(path) as shard:
                depths = shard["depth"]
                masks = shard["object_mask"]
                complete_points = shard["points_camera"]
                for sample_id in range(self.counts[shard_id]):
                    global_id = start + sample_id
                    partial, complete_estimated, complete_object, _, _ = prepare_two_stage_clouds(
                        depths[sample_id], masks[sample_id], complete_points[sample_id],
                        shard["pose_camera"][sample_id], self.intrinsics,
                        num_points, rng, self.max_depth,
                    )
                    self.partial_clouds[global_id] = partial
                    self.complete_estimated[global_id] = complete_estimated
                    self.complete_object[global_id] = complete_object
            print("Prepared paired clouds from {} ({}/{})".format(
                path.name, shard_id + 1, len(self.shard_paths)
            ))

    def __len__(self):
        return len(self.partial_clouds)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.partial_clouds[index]),
            torch.from_numpy(self.complete_estimated[index]),
            torch.from_numpy(self.complete_object[index]),
        )


def build_pointr_model(device, checkpoint=None):
    """Build the official PoinTr model and optionally load a checkpoint."""
    if device.type != "cuda":
        raise RuntimeError(
            "The installed PoinTr/PointNet++ extensions are configured for CUDA; "
            "please train with --device cuda or cuda:0."
        )
    if not POINTR_CONFIG.is_file():
        raise FileNotFoundError("PoinTr config not found: {}".format(POINTR_CONFIG))

    sys.path.insert(0, str(POINTR_ROOT))
    previous_directory = os.getcwd()
    try:
        # PoinTr's YAML includes cfgs/dataset_configs/PCN.yaml relative to its root.
        os.chdir(POINTR_ROOT)
        from tools import builder
        from utils.config import cfg_from_yaml_file

        config = cfg_from_yaml_file(str(POINTR_CONFIG))
        model = builder.model_builder(config.model)
    finally:
        os.chdir(previous_directory)

    if checkpoint is not None:
        builder.load_model(model, str(checkpoint))
    model.to(device)
    return model


def sample_points_fps(points, num_points):
    """Downsample a dense PoinTr output while preserving point-coordinate gradients."""
    sys.path.insert(0, str(POINTR_ROOT))
    from pointnet2_ops import pointnet2_utils

    indices = pointnet2_utils.furthest_point_sample(points.contiguous(), num_points)
    return pointnet2_utils.gather_operation(
        points.transpose(1, 2).contiguous(), indices,
    ).transpose(1, 2).contiguous()


def _run_epoch(stage1, stage2, chamfer_distance, loader, optimizer, device, num_points):
    training = optimizer is not None
    stage1.train(training)
    stage2.train(training)
    totals = np.zeros(4, dtype=np.float64)
    samples = 0

    with torch.set_grad_enabled(training):
        for partial, complete_estimated, complete_object in loader:
            partial = partial.to(device, non_blocking=True)
            complete_estimated = complete_estimated.to(device, non_blocking=True)
            complete_object = complete_object.to(device, non_blocking=True)
            stage1_ret = stage1(partial)
            stage2_input = sample_points_fps(stage1_ret[-1], num_points)
            stage2_ret = stage2(stage2_input)
            # PoinTr appends its input points to both outputs for same-frame completion.
            # Stage 2 changes frames, so supervise only the generated canonical points.
            stage2_coarse = stage2_ret[0][:, :stage2.num_query]
            stage2_fine = stage2_ret[-1][:, :stage2.num_pred]
            losses = (
                chamfer_distance(stage1_ret[0], complete_estimated),
                chamfer_distance(stage1_ret[-1], complete_estimated),
                chamfer_distance(stage2_coarse, complete_object),
                chamfer_distance(stage2_fine, complete_object),
            )
            loss = sum(losses)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(stage1.parameters()) + list(stage2.parameters()), 10.0,
                )
                optimizer.step()

            batch_size = partial.shape[0]
            totals += np.asarray([value.detach().item() for value in losses]) * batch_size
            samples += batch_size

    denominator = max(samples, 1)
    return tuple((totals / denominator).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/objects")
    parser.add_argument("--output", default="runs/rgbd_box_estimator/pointr_two_stage_best.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Two PoinTr models are trained jointly; reduce this if GPU memory is limited.",
    )
    parser.add_argument(
        "--num-points", type=int, default=512,
        help="sampled points per partial/full pair; preserves existing preprocessing",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--pretrained", default=str(DEFAULT_PRETRAINED))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.num_points < 512:
        raise ValueError("epochs and batch size must be positive; PoinTr requires at least 512 points")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay cannot be negative")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")

    dataset_dir = Path(args.dataset)
    shard_paths = sorted(dataset_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError("No shard_*.npz files found in {}".format(dataset_dir))
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Camera intrinsics metadata not found: {}".format(metadata_path))

    metadata = json.loads(metadata_path.read_text())
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = RGBDPointCloudDataset(
        shard_paths, intrinsics, num_points=args.num_points,
        max_depth=args.max_depth, seed=args.seed,
    )
    total = len(dataset)
    if total < 2:
        raise ValueError("At least two samples are required for train/validation split")

    rng = np.random.default_rng(args.seed)
    all_indices = rng.permutation(total)
    val_count = min(max(1, int(total * args.val_fraction)), total - 1)
    val_indices, train_indices = all_indices[:val_count], all_indices[val_count:]

    device = torch.device(args.device)
    pretrained = Path(args.pretrained) if args.pretrained else None
    if pretrained is not None and not pretrained.is_file():
        raise FileNotFoundError("Pretrained PoinTr checkpoint not found: {}".format(pretrained))

    stage1 = build_pointr_model(device, pretrained)
    stage2 = build_pointr_model(device, pretrained)
    sys.path.insert(0, str(POINTR_ROOT))
    from extensions.chamfer_dist import ChamferDistanceL1

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()), batch_size=args.batch_size,
        shuffle=True, num_workers=0, drop_last=False,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()), batch_size=args.batch_size,
        shuffle=False, num_workers=0, drop_last=False,
        pin_memory=device.type == "cuda",
    )
    chamfer_distance = ChamferDistanceL1()
    optimizer = torch.optim.AdamW(
        list(stage1.parameters()) + list(stage2.parameters()),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_dense = float("inf")

    print("Jointly training two PoinTr models from checkpoint: {}".format(pretrained), flush=True)
    print("samples: {}; train: {}; validation: {}; points: {}; device: {}".format(
        total, len(train_indices), len(val_indices), args.num_points, device,
    ), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_losses = _run_epoch(
            stage1, stage2, chamfer_distance, train_loader, optimizer, device,
            args.num_points,
        )
        with torch.no_grad():
            val_losses = _run_epoch(
                stage1, stage2, chamfer_distance, val_loader, None, device,
                args.num_points,
            )
        print(
            "epoch {:03d}/{:03d}: "
            "train stage1 coarse={:.6f}, fine={:.6f}; stage2 coarse={:.6f}, fine={:.6f}; "
            "val stage1 coarse={:.6f}, fine={:.6f}; stage2 coarse={:.6f}, fine={:.6f}".format(
                epoch, args.epochs, *train_losses, *val_losses,
            ),
            flush=True,
        )

        if val_losses[3] < best_dense:
            best_dense = val_losses[3]
            torch.save({
                "stage1_model": stage1.state_dict(),
                "stage2_model": stage2.state_dict(),
                "model_type": "pointr_rgbd_two_stage",
                "num_points": args.num_points,
                "intrinsics": intrinsics,
                "max_depth": args.max_depth,
                "epoch": epoch,
                "val_stage1_coarse_loss": val_losses[0],
                "val_stage1_fine_loss": val_losses[1],
                "val_stage2_coarse_loss": val_losses[2],
                "val_stage2_fine_loss": val_losses[3],
                "pretrained": str(pretrained),
                "input_frame_method": "partial-cloud centroid; image-up z; horizontal PCA x",
                "stage1_target_frame": "partial-cloud estimated frame",
                "stage2_target_frame": "object asset root, supervised by pose_camera",
            }, output_path)
            print("saved best checkpoint: {}".format(output_path), flush=True)


if __name__ == "__main__":
    main()
