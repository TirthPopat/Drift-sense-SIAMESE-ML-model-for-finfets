#!/usr/bin/env python3
"""
Evaluate the trained Siamese model on a dataset split.

Reports:
  - Euclidean localization error: mean, median, worst-case
  - Pass rates at 5, 4, 2, 1 pixel thresholds
  - Runtime per image pair
  - Comparison with ZNCC baseline (if available)
  - Failure case visualization

Example:
    python model/evaluate_model.py --manifest ./output/val/manifest.csv --checkpoint model/checkpoints/best.pt
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
from model.dataset import DriftSenseDataset


def evaluate(model, dataset, device, output_dir=None):
    """Run evaluation, return per-sample results and aggregate metrics."""
    model.eval()

    results = []
    total_time_ms = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        template = sample["template"].unsqueeze(0).to(device)
        search = sample["search"].unsqueeze(0).to(device)
        gt_x = sample["gt_x"].item()
        gt_y = sample["gt_y"].item()

        t0 = time.time()
        with torch.no_grad():
            output = model(template, search)
        elapsed_ms = (time.time() - t0) * 1000
        total_time_ms += elapsed_ms

        pred_x = output["pred_x"].item()
        pred_y = output["pred_y"].item()

        # Clamp to valid range
        pred_x = max(0, min(pred_x, 1000))
        pred_y = max(0, min(pred_y, 1000))

        dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        results.append({
            "id": sample["id"],
            "architecture": sample["architecture"],
            "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": pred_x, "pred_y": pred_y,
            "distance_px": dist,
            "time_ms": elapsed_ms,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(dataset)}] avg dist so far: "
                  f"{np.mean([r['distance_px'] for r in results]):.2f} px")

    # Aggregate metrics
    distances = np.array([r["distance_px"] for r in results])
    times = np.array([r["time_ms"] for r in results])

    metrics = {
        "n_samples": len(results),
        "mean_dist": distances.mean(),
        "median_dist": np.median(distances),
        "std_dist": distances.std(),
        "max_dist": distances.max(),
        "min_dist": distances.min(),
        "pass_5px": (distances <= 5.0).mean(),
        "pass_4px": (distances <= 4.0).mean(),
        "pass_2px": (distances <= 2.0).mean(),
        "pass_1px": (distances <= 1.0).mean(),
        "mean_time_ms": times.mean(),
        "total_time_s": times.sum() / 1000,
    }

    return results, metrics


def run_zncc_baseline(dataset):
    """Run ZNCC baseline for comparison (if available)."""
    try:
        from baseline_solution.zncc import zncc_match
    except ImportError:
        print("  ZNCC baseline not available, skipping comparison.")
        return None

    results = []
    for i in range(len(dataset)):
        sample = dataset[i]
        ref = (sample["template"].squeeze().numpy() * 255).astype(np.uint8)
        search = (sample["search"].squeeze().numpy() * 255).astype(np.uint8)
        gt_x = sample["gt_x"].item()
        gt_y = sample["gt_y"].item()

        # ZNCC expects full-res reference, not downsampled template
        # Re-upscale for fair comparison (ZNCC does its own downsampling)
        ref_fullres = cv2.resize(ref, (1000, 1000), interpolation=cv2.INTER_CUBIC)

        t0 = time.time()
        match = zncc_match(ref_fullres, search)
        elapsed_ms = (time.time() - t0) * 1000

        dist = np.sqrt((match["x"] - gt_x) ** 2 + (match["y"] - gt_y) ** 2)
        results.append({
            "id": sample["id"],
            "pred_x": match["x"], "pred_y": match["y"],
            "distance_px": dist,
            "time_ms": elapsed_ms,
        })

    distances = np.array([r["distance_px"] for r in results])
    times = np.array([r["time_ms"] for r in results])
    return {
        "mean_dist": distances.mean(),
        "median_dist": np.median(distances),
        "pass_5px": (distances <= 5.0).mean(),
        "pass_2px": (distances <= 2.0).mean(),
        "pass_1px": (distances <= 1.0).mean(),
        "mean_time_ms": times.mean(),
    }


def save_failure_cases(results, dataset, output_dir, top_k=5):
    """Save visualizations of the worst predictions."""
    failure_dir = os.path.join(output_dir, "failure_cases")
    os.makedirs(failure_dir, exist_ok=True)

    # Sort by distance (worst first)
    sorted_results = sorted(results, key=lambda r: r["distance_px"], reverse=True)

    for rank, r in enumerate(sorted_results[:top_k]):
        idx = next(i for i in range(len(dataset)) if dataset[i]["id"] == r["id"])
        sample = dataset[idx]
        search = (sample["search"].squeeze().numpy() * 255).astype(np.uint8)
        search_bgr = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

        # Draw ground truth (green)
        gt_x, gt_y = int(round(r["gt_x"])), int(round(r["gt_y"]))
        cv2.drawMarker(search_bgr, (gt_x, gt_y), (0, 255, 0),
                        markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # Draw prediction (red)
        px, py = int(round(r["pred_x"])), int(round(r["pred_y"]))
        cv2.drawMarker(search_bgr, (px, py), (0, 0, 255),
                        markerType=cv2.MARKER_TILTED_CROSS, markerSize=20, thickness=2)

        # Draw line between them
        cv2.line(search_bgr, (gt_x, gt_y), (px, py), (255, 255, 0), 1)

        # Add text
        cv2.putText(search_bgr,
                     f"#{rank+1} dist={r['distance_px']:.1f}px arch={r['architecture']}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        path = os.path.join(failure_dir, f"failure_{rank+1:02d}_dist{r['distance_px']:.1f}.png")
        cv2.imwrite(path, search_bgr)

    print(f"  Saved top-{top_k} failure cases to {failure_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="./output/val/manifest.csv")
    parser.add_argument("--checkpoint", default="./model/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compare-zncc", action="store_true", help="Also run ZNCC baseline for comparison")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = SiameseCorrelationNet(pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Load dataset
    dataset = DriftSenseDataset(args.manifest, max_samples=args.max_samples, augment=False)
    print(f"Evaluating on {len(dataset)} samples...")

    # Evaluate
    results, metrics = evaluate(model, dataset, device, args.output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  DL MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples:          {metrics['n_samples']}")
    print(f"  Mean distance:    {metrics['mean_dist']:.2f} px")
    print(f"  Median distance:  {metrics['median_dist']:.2f} px")
    print(f"  Std distance:     {metrics['std_dist']:.2f} px")
    print(f"  Worst distance:   {metrics['max_dist']:.2f} px")
    print(f"  Best distance:    {metrics['min_dist']:.2f} px")
    print(f"  Pass @5px:        {metrics['pass_5px']:.1%}")
    print(f"  Pass @4px:        {metrics['pass_4px']:.1%}")
    print(f"  Pass @2px:        {metrics['pass_2px']:.1%}")
    print(f"  Pass @1px:        {metrics['pass_1px']:.1%}")
    print(f"  Mean time/pair:   {metrics['mean_time_ms']:.1f} ms")
    print(f"  Total eval time:  {metrics['total_time_s']:.1f} s")

    # Save per-sample results CSV
    results_path = os.path.join(args.output_dir, "predictions.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Predictions saved to {results_path}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"  Summary saved to {summary_path}")

    # Save failure cases
    save_failure_cases(results, dataset, args.output_dir)

    # Optional ZNCC comparison
    if args.compare_zncc:
        print(f"\nRunning ZNCC baseline for comparison...")
        zncc_metrics = run_zncc_baseline(dataset)
        if zncc_metrics:
            print(f"\n{'='*60}")
            print(f"  COMPARISON: DL Model vs ZNCC Baseline")
            print(f"{'='*60}")
            print(f"  {'Metric':<20} {'DL Model':>12} {'ZNCC':>12}")
            print(f"  {'-'*44}")
            print(f"  {'Mean dist (px)':<20} {metrics['mean_dist']:>11.2f} {zncc_metrics['mean_dist']:>11.2f}")
            print(f"  {'Median dist (px)':<20} {metrics['median_dist']:>11.2f} {zncc_metrics['median_dist']:>11.2f}")
            print(f"  {'Pass @5px':<20} {metrics['pass_5px']:>11.1%} {zncc_metrics['pass_5px']:>11.1%}")
            print(f"  {'Pass @2px':<20} {metrics['pass_2px']:>11.1%} {zncc_metrics['pass_2px']:>11.1%}")
            print(f"  {'Pass @1px':<20} {metrics['pass_1px']:>11.1%} {zncc_metrics['pass_1px']:>11.1%}")
            print(f"  {'Time/pair (ms)':<20} {metrics['mean_time_ms']:>11.1f} {zncc_metrics['mean_time_ms']:>11.1f}")


if __name__ == "__main__":
    main()
