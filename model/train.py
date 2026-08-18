#!/usr/bin/env python3
"""
Train the Siamese Correlation Network for Drift-Sense localization.

Uses a dual loss:
  1. Coordinate MSE: direct regression on (pred_x, pred_y) vs (gt_x, gt_y)
  2. Heatmap BCE: ground-truth Gaussian heatmap vs predicted correlation map

The heatmap loss teaches the network WHERE to look (spatial structure),
while the coordinate loss teaches it to be precise (sub-pixel accuracy).

Example:
    python model/train.py --epochs 50 --batch-size 4 --lr 1e-3
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.siamese_model import SiameseCorrelationNet, pixel_to_corr, make_gt_heatmap
from model.dataset import create_dataloaders
import numpy as np


def compute_metrics(pred_x, pred_y, gt_x, gt_y):
    """Compute Euclidean distance and pass rates at standard thresholds."""
    dist = torch.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
    metrics = {
        "mean_dist": dist.mean().item(),
        "median_dist": dist.median().item(),
        "max_dist": dist.max().item(),
        "pass_5px": (dist <= 5.0).float().mean().item(),
        "pass_4px": (dist <= 4.0).float().mean().item(),
        "pass_2px": (dist <= 2.0).float().mean().item(),
        "pass_1px": (dist <= 1.0).float().mean().item(),
    }
    return metrics


def train_one_epoch(model, loader, optimizer, device, heatmap_weight=1.0):
    model.train()
    total_loss = 0
    total_coord_loss = 0
    total_heatmap_loss = 0
    all_pred_x, all_pred_y, all_gt_x, all_gt_y = [], [], [], []

    for batch in tqdm(loader, desc="  train", leave=False):
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_x = batch["gt_x"].to(device)
        gt_y = batch["gt_y"].to(device)

        output = model(template, search)
        pred_x = output["pred_x"]
        pred_y = output["pred_y"]
        heatmap = output["heatmap"]

        # Loss 1: Coordinate MSE
        coord_loss = F.mse_loss(pred_x, gt_x) + F.mse_loss(pred_y, gt_y)

        # Loss 2: Heatmap supervision
        gt_corr_x, gt_corr_y = pixel_to_corr(gt_x, gt_y)
        _, _, H, W = heatmap.shape
        gt_heatmap = make_gt_heatmap(
            gt_corr_x, gt_corr_y, H, W, sigma=2.0, device=device
        )
        # Normalize predicted heatmap for comparison
        heatmap_pred = torch.sigmoid(heatmap)
        heatmap_loss = F.mse_loss(heatmap_pred, gt_heatmap)

        loss = coord_loss + heatmap_weight * heatmap_loss

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * template.size(0)
        total_coord_loss += coord_loss.item() * template.size(0)
        total_heatmap_loss += heatmap_loss.item() * template.size(0)

        all_pred_x.append(pred_x.detach())
        all_pred_y.append(pred_y.detach())
        all_gt_x.append(gt_x.detach())
        all_gt_y.append(gt_y.detach())

    n = len(loader.dataset)
    all_pred_x = torch.cat(all_pred_x)
    all_pred_y = torch.cat(all_pred_y)
    all_gt_x = torch.cat(all_gt_x)
    all_gt_y = torch.cat(all_gt_y)

    metrics = compute_metrics(all_pred_x, all_pred_y, all_gt_x, all_gt_y)
    metrics["loss"] = total_loss / n
    metrics["coord_loss"] = total_coord_loss / n
    metrics["heatmap_loss"] = total_heatmap_loss / n
    return metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    all_pred_x, all_pred_y, all_gt_x, all_gt_y = [], [], [], []

    for batch in tqdm(loader, desc="  val", leave=False):
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_x = batch["gt_x"].to(device)
        gt_y = batch["gt_y"].to(device)

        output = model(template, search)
        pred_x = output["pred_x"]
        pred_y = output["pred_y"]

        coord_loss = F.mse_loss(pred_x, gt_x) + F.mse_loss(pred_y, gt_y)
        total_loss += coord_loss.item() * template.size(0)

        all_pred_x.append(pred_x)
        all_pred_y.append(pred_y)
        all_gt_x.append(gt_x)
        all_gt_y.append(gt_y)

    n = len(loader.dataset)
    all_pred_x = torch.cat(all_pred_x)
    all_pred_y = torch.cat(all_pred_y)
    all_gt_x = torch.cat(all_gt_x)
    all_gt_y = torch.cat(all_gt_y)

    metrics = compute_metrics(all_pred_x, all_pred_y, all_gt_x, all_gt_y)
    metrics["loss"] = total_loss / n
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-manifest", default="./output/train/manifest.csv")
    parser.add_argument("--val-manifest", default="./output/val/manifest.csv")
    parser.add_argument("--output-dir", default="./model/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--heatmap-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data
    train_loader, val_loader = create_dataloaders(
        args.train_manifest, args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # Model
    model = SiameseCorrelationNet(pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} params ({trainable_params:,} trainable)")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_dist = float("inf")
    patience_counter = 0
    history = []

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Mean':>8} | "
          f"{'@5px':>6} | {'@2px':>6} | {'@1px':>6} | {'LR':>8} | {'Time':>5}")
    print("-" * 85)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            heatmap_weight=args.heatmap_weight,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"{epoch:5d} | {train_metrics['loss']:10.4f} | {val_metrics['loss']:8.4f} | "
            f"{val_metrics['mean_dist']:7.2f}px | "
            f"{val_metrics['pass_5px']:5.1%} | {val_metrics['pass_2px']:5.1%} | "
            f"{val_metrics['pass_1px']:5.1%} | {lr:8.6f} | {elapsed:4.0f}s"
        )

        # Save history
        record = {"epoch": epoch, "lr": lr, "elapsed": elapsed}
        record.update({f"train_{k}": v for k, v in train_metrics.items()})
        record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(record)

        # Save best model (by val mean Euclidean distance)
        if val_metrics["mean_dist"] < best_val_dist:
            best_val_dist = val_metrics["mean_dist"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
            }, os.path.join(args.output_dir, "best.pt"))
            print(f"       * saved best model (val mean dist: {best_val_dist:.2f} px)")
        else:
            patience_counter += 1

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(args.output_dir, "latest.pt"))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Save training history
    import json
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val mean distance: {best_val_dist:.2f} px")
    print(f"Checkpoint saved to: {os.path.join(args.output_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
