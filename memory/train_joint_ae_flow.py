"""Jointly train the trajectory AutoEncoder and its RealNVP latent prior.

The optimized objective is::

    reconstruction_loss + flow_loss_weight * latent_flow_nll

The trajectory input is the original initial-object-frame representation. It
writes ``ae_best.pt`` and ``flow_best.pt`` in a format accepted by the existing
replay/search scripts, plus ``joint_best.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from memory.train_latent_flow import RealNVP, _standard_normal_log_prob
from memory.train_trajectory_ae import (
    TemporalAutoEncoder,
    _load_data,
    _loss,
)


def _make_batch(args, indices, trajectories):
    return trajectories[indices].to(args.device)


def _configs(args, model, flow, coordinate_frame):
    ae_config = {
        "input_dim": 15,
        "sequence_length": args.sequence_length,
        "num_tokens": args.num_tokens,
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "rotation_matrix_loss_weight": args.rotation_matrix_loss_weight,
        "feature_order": ["wrist_position_object", "wrist_rotation6d_object", "raw_hand_joints"],
        "normalization": "none; wrist position is relative to the initial object frame",
        "coordinate_frame": coordinate_frame,
    }
    flow_config = {
        "latent_dim": model.num_tokens * model.latent_dim,
        "latent_shape": [model.num_tokens, model.latent_dim],
        "layers": args.flow_layers,
        "hidden_dim": args.flow_hidden_dim,
        "scale_limit": args.flow_scale_limit,
        "base_distribution": "standard_normal",
        "coordinate_frame": coordinate_frame,
    }
    return ae_config, flow_config


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trajectories = _load_data(args.dataset)
    sample_count = trajectories.shape[0]
    coordinate_frame = "initial_object"
    sequence_length = trajectories.shape[1]
    if sequence_length != args.sequence_length:
        raise ValueError(f"Expected {args.sequence_length} steps, got {sequence_length}")
    if sample_count < 2:
        raise ValueError("At least two trajectories are required")

    permutation = torch.randperm(sample_count)
    split = max(1, int(sample_count * (1.0 - args.validation_fraction)))
    train_indices, valid_indices = permutation[:split], permutation[split:]
    if valid_indices.numel() == 0:
        valid_indices = train_indices[:1]
    model = TemporalAutoEncoder(15, args.sequence_length, args.num_tokens, args.latent_dim, args.hidden_dim).to(args.device)
    loader = DataLoader(train_indices, batch_size=args.batch_size, shuffle=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ae_optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_ae = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train(); train_values = []
        for indices in loader:
            batch = _make_batch(args, indices.long(), trajectories)
            reconstruction, _ = model(batch)
            ae_total, _, _ = _loss(batch, reconstruction, args.rotation_matrix_loss_weight)
            ae_optimizer.zero_grad(set_to_none=True); ae_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm); ae_optimizer.step()
            train_values.append(float(ae_total.detach().cpu()))

        model.eval(); val_values = []
        with torch.no_grad():
            for start in range(0, valid_indices.numel(), args.batch_size):
                batch = _make_batch(args, valid_indices[start:start + args.batch_size].long(), trajectories)
                reconstruction, _ = model(batch)
                ae_total, _, _ = _loss(batch, reconstruction, args.rotation_matrix_loss_weight)
                val_values.append(float(ae_total.cpu()))
        validation = float(np.mean(val_values))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"AE epoch {epoch:04d}/{args.epochs}: train={np.mean(train_values):.6f} val={validation:.6f}")
        if validation < best_ae:
            best_ae = validation
            ae_config, _ = _configs(args, model, model, coordinate_frame)
            torch.save({"model": model.state_dict(), "config": ae_config, "epoch": epoch, "validation_loss": validation}, output_dir / "ae_best.pt")

    # Freeze the finished AE and train the flow on a fixed latent collection.
    ae_checkpoint = torch.load(output_dir / "ae_best.pt", map_location=args.device)
    model.load_state_dict(ae_checkpoint["model"]); model.eval()
    flow_dim = args.num_tokens * args.latent_dim
    flow = RealNVP(flow_dim, args.flow_layers, args.flow_hidden_dim, args.flow_scale_limit).to(args.device)
    with torch.no_grad():
        latent_parts = []
        for start in range(0, train_indices.numel(), args.batch_size):
            batch = _make_batch(args, train_indices[start:start + args.batch_size].long(), trajectories)
            latent_parts.append(model.encode(batch).reshape(batch.shape[0], -1).cpu())
        valid_latent_parts = []
        for start in range(0, valid_indices.numel(), args.batch_size):
            batch = _make_batch(args, valid_indices[start:start + args.batch_size].long(), trajectories)
            valid_latent_parts.append(model.encode(batch).reshape(batch.shape[0], -1).cpu())
    train_latents = torch.cat(latent_parts); valid_latents = torch.cat(valid_latent_parts)
    flow_optimizer = torch.optim.Adam(flow.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    flow_loader = DataLoader(train_latents, batch_size=args.batch_size, shuffle=True)
    best_flow = math.inf
    for epoch in range(1, args.epochs + 1):
        flow.train(); train_nll = []
        for batch in flow_loader:
            gaussian, log_det = flow(batch.to(args.device))
            loss = -(_standard_normal_log_prob(gaussian) + log_det).mean()
            flow_optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), args.max_grad_norm); flow_optimizer.step()
            train_nll.append(float(loss.detach().cpu()))
        flow.eval()
        with torch.no_grad():
            gaussian, log_det = flow(valid_latents.to(args.device))
            validation = float(-(_standard_normal_log_prob(gaussian) + log_det).mean().cpu())
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Flow epoch {epoch:04d}/{args.epochs}: train_nll={np.mean(train_nll):.6f} val_nll={validation:.6f}")
        if validation < best_flow:
            best_flow = validation
            _, flow_config = _configs(args, model, flow, coordinate_frame)
            torch.save({"model": flow.state_dict(), "config": flow_config, "epoch": epoch, "validation_loss": validation}, output_dir / "flow_best.pt")
            torch.save({"ae_model": model.state_dict(), "flow_model": flow.state_dict(), "ae_config": ae_config, "flow_config": flow_config, "epoch": epoch, "validation_loss": validation}, output_dir / "joint_best.pt")
    print(f"Saved joint checkpoints to {output_dir}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True); p.add_argument("--output-dir", default="runs/joint_ae_flow")
    p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--sequence-length", type=int, default=20); p.add_argument("--num-tokens", type=int, default=5)
    p.add_argument("--latent-dim", type=int, default=6); p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--rotation-matrix-loss-weight", type=float, default=1.0)
    p.add_argument("--flow-layers", type=int, default=6); p.add_argument("--flow-hidden-dim", type=int, default=128)
    p.add_argument("--flow-scale-limit", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=1e-4); p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=5.0); p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--log-every", type=int, default=10)
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
