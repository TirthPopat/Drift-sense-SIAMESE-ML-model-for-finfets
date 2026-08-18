import os
import argparse
import random
import pandas as pd
import numpy as np
import torch
from model.hybrid_localize import load_model, hybrid_predict

def main():
    parser = argparse.ArgumentParser(description="Live Demo Inference on Curated Samples")
    parser.add_argument("--manifest", default="dataset_curated/manifest.csv")
    parser.add_argument("--checkpoint", default="model/checkpoints/best.pt")
    parser.add_argument("--id", type=int, default=None, help="Specific Sample ID to evaluate")
    parser.add_argument("--tier", default="no_noise", choices=["no_noise", "low_noise", "high_noise", "failure_case", "all"])
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"Error: Manifest not found at {args.manifest}")
        return

    df = pd.read_csv(args.manifest)
    if args.tier != "all":
        df = df[df["tier"] == args.tier]

    if args.id is not None:
        sample_df = df[df["id"] == args.id]
        if len(sample_df) == 0:
            print(f"Sample ID {args.id} not found in tier {args.tier}")
            return
        sample = sample_df.iloc[0]
    else:
        sample = df.iloc[random.randint(0, len(df) - 1)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    ref_path = sample["reference_path"]
    search_path = sample["search_path"]
    gt_x = float(sample["gt_x"])
    gt_y = float(sample["gt_y"])
    arch = sample["architecture"]
    seed = sample.get("seed", 42)
    tier = sample.get("tier", "unknown")

    pred_x, pred_y, elapsed_ms, cx, cy = hybrid_predict(
        model, ref_path, search_path, device, window_radius=80
    )

    dx = pred_x - gt_x
    dy = pred_y - gt_y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    print("\n" + "=" * 85)
    print(f"Ground truth center in search image: ({gt_x:.1f}, {gt_y:.1f}) px | Architecture: {arch} | Seed: {seed} | Tier: {tier}")
    print("-" * 85)
    print(f"[*] Run Hybrid Matcher (ResNet-18 + Rotated ZNCC) on this sample")
    print(f"Hybrid prediction: ({pred_x:.2f}, {pred_y:.2f}) | shift (predicted - actual): dx={dx:+.2f} dy={dy:+.2f} | distance: {dist:.2f} px | Latency: {elapsed_ms:.1f} ms")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()

