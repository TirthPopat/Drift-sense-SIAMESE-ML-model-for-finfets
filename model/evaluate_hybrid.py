#!/usr/bin/env python3
"""
Evaluate the Hybrid (DL + ZNCC) pipeline on a dataset split.

Compares three approaches side-by-side:
  1. DL-only (coarse Siamese model)
  2. ZNCC-only (baseline template matching)
  3. Hybrid (DL coarse + ZNCC fine refinement)

Example:
    python model/evaluate_hybrid.py --manifest ./output/val_finfet/manifest.csv
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.siamese_model import SiameseCorrelationNet
from model.hybrid_localize import hybrid_predict, load_model


def zncc_only_predict(reference_path, search_path):
    """Run standalone multi-scale ZNCC (the baseline approach)."""
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        return None, None, 0

    h_ref, w_ref = reference.shape
    scales = [9.0, 9.5, 10.0, 10.5, 11.0]
    best_score = -1.0
    best_x, best_y = 500.0, 500.0

    t0 = time.time()
    for scale in scales:
        new_w = max(1, int(round(w_ref / scale)))
        new_h = max(1, int(round(h_ref / scale)))
        template = cv2.resize(reference, (new_w, new_h),
                              interpolation=cv2.INTER_AREA)
        th, tw = template.shape[:2]

        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = max_val
            best_x = max_loc[0] + tw / 2.0
            best_y = max_loc[1] + th / 2.0

    elapsed_ms = (time.time() - t0) * 1000
    return best_x, best_y, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="./output/val_finfet/manifest.csv")
    parser.add_argument("--checkpoint", default="./model/checkpoints/best_finfet.pt")
    parser.add_argument("--output-dir", default="./results_finfet")
    parser.add_argument("--window-radius", type=int, default=40)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load manifest
    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"Evaluating on {len(rows)} samples...")

    # Resolve paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = []
    for i, row in enumerate(rows):
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        ref_path = row["reference_path"]
        search_path = row["search_path"]
        if not os.path.isabs(ref_path):
            if ref_path.startswith("./") or ref_path.startswith(".\\"):
                ref_path = ref_path[2:]
            if search_path.startswith("./") or search_path.startswith(".\\"):
                search_path = search_path[2:]
            ref_path = os.path.join(project_root, ref_path)
            search_path = os.path.join(project_root, search_path)

        # --- Hybrid prediction ---
        hybrid_x, hybrid_y, hybrid_ms, coarse_x, coarse_y = hybrid_predict(
            model, ref_path, search_path, device,
            window_radius=args.window_radius,
        )
        hybrid_dist = np.sqrt((hybrid_x - gt_x)**2 + (hybrid_y - gt_y)**2)
        coarse_dist = np.sqrt((coarse_x - gt_x)**2 + (coarse_y - gt_y)**2)

        # --- ZNCC-only prediction ---
        zncc_x, zncc_y, zncc_ms = zncc_only_predict(ref_path, search_path)
        zncc_dist = np.sqrt((zncc_x - gt_x)**2 + (zncc_y - gt_y)**2) if zncc_x else float("inf")

        results.append({
            "id": row["id"],
            "architecture": row["architecture"],
            "gt_x": gt_x, "gt_y": gt_y,
            "coarse_x": coarse_x, "coarse_y": coarse_y, "coarse_dist": coarse_dist,
            "hybrid_x": hybrid_x, "hybrid_y": hybrid_y, "hybrid_dist": hybrid_dist,
            "hybrid_ms": hybrid_ms,
            "zncc_x": zncc_x, "zncc_y": zncc_y, "zncc_dist": zncc_dist,
            "zncc_ms": zncc_ms,
        })

        if (i + 1) % 50 == 0:
            h_dists = [r["hybrid_dist"] for r in results]
            print(f"  [{i+1}/{len(rows)}] hybrid avg: {np.mean(h_dists):.2f}px, "
                  f"pass@5px: {np.mean(np.array(h_dists) <= 5):.1%}")

    # Aggregate
    hybrid_dists = np.array([r["hybrid_dist"] for r in results])
    coarse_dists = np.array([r["coarse_dist"] for r in results])
    zncc_dists = np.array([r["zncc_dist"] for r in results])
    hybrid_times = np.array([r["hybrid_ms"] for r in results])
    zncc_times = np.array([r["zncc_ms"] for r in results])

    print(f"\n{'='*70}")
    print(f"  THREE-WAY COMPARISON: DL-Only vs ZNCC-Only vs Hybrid")
    print(f"{'='*70}")
    print(f"  {'Metric':<22} {'DL-Only':>10} {'ZNCC-Only':>10} {'Hybrid':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Mean dist (px)':<22} {coarse_dists.mean():>9.2f} {zncc_dists.mean():>9.2f} {hybrid_dists.mean():>9.2f}")
    print(f"  {'Median dist (px)':<22} {np.median(coarse_dists):>9.2f} {np.median(zncc_dists):>9.2f} {np.median(hybrid_dists):>9.2f}")
    print(f"  {'Pass @5px':<22} {(coarse_dists<=5).mean():>9.1%} {(zncc_dists<=5).mean():>9.1%} {(hybrid_dists<=5).mean():>9.1%}")
    print(f"  {'Pass @4px':<22} {(coarse_dists<=4).mean():>9.1%} {(zncc_dists<=4).mean():>9.1%} {(hybrid_dists<=4).mean():>9.1%}")
    print(f"  {'Pass @2px':<22} {(coarse_dists<=2).mean():>9.1%} {(zncc_dists<=2).mean():>9.1%} {(hybrid_dists<=2).mean():>9.1%}")
    print(f"  {'Pass @1px':<22} {(coarse_dists<=1).mean():>9.1%} {(zncc_dists<=1).mean():>9.1%} {(hybrid_dists<=1).mean():>9.1%}")
    print(f"  {'Time/pair (ms)':<22} {'N/A':>10} {zncc_times.mean():>9.1f} {hybrid_times.mean():>9.1f}")
    print(f"  {'Worst dist (px)':<22} {coarse_dists.max():>9.2f} {zncc_dists.max():>9.2f} {hybrid_dists.max():>9.2f}")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "hybrid_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved predictions to {csv_path}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "hybrid_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Hybrid Pass @5px: {(hybrid_dists<=5).mean():.1%}\n")
        f.write(f"Hybrid Pass @2px: {(hybrid_dists<=2).mean():.1%}\n")
        f.write(f"Hybrid Pass @1px: {(hybrid_dists<=1).mean():.1%}\n")
        f.write(f"Hybrid Mean dist: {hybrid_dists.mean():.2f}\n")
        f.write(f"Hybrid Median dist: {np.median(hybrid_dists):.2f}\n")
        f.write(f"Hybrid Mean time: {hybrid_times.mean():.1f} ms\n")
    print(f"  Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
