#!/usr/bin/env python3
"""
Standalone inference script for Drift-Sense localization.

Matches the exact submission interface Applied Materials requires:
accepts a reference and search image path, prints predicted (x, y).

Example:
    python model/localize.py --reference ref.png --search search.png
    # Output: 450.32,550.17

    python model/localize.py --reference ref.png --search search.png --checkpoint model/checkpoints/best.pt
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.siamese_model import SiameseCorrelationNet


def load_model(checkpoint_path: str, device: torch.device) -> SiameseCorrelationNet:
    """Load trained model from checkpoint."""
    model = SiameseCorrelationNet(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(
    model: SiameseCorrelationNet,
    reference_path: str,
    search_path: str,
    device: torch.device,
) -> tuple:
    """Run inference on a single image pair.

    Returns (pred_x, pred_y, elapsed_ms).
    """
    # Load images
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Cannot read reference: {reference_path}")
    if search is None:
        raise FileNotFoundError(f"Cannot read search: {search_path}")

    # Downsample reference 10x
    h, w = reference.shape
    template = cv2.resize(reference, (w // 10, h // 10), interpolation=cv2.INTER_AREA)

    # To tensors
    template_t = torch.from_numpy(template.astype(np.float32) / 255.0)
    template_t = template_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 100, 100)
    search_t = torch.from_numpy(search.astype(np.float32) / 255.0)
    search_t = search_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 1000, 1000)

    # Inference
    t0 = time.time()
    with torch.no_grad():
        output = model(template_t, search_t)
    elapsed_ms = (time.time() - t0) * 1000

    pred_x = output["pred_x"].item()
    pred_y = output["pred_y"].item()

    # Clamp to valid range
    pred_x = max(0, min(pred_x, search.shape[1]))
    pred_y = max(0, min(pred_y, search.shape[0]))

    return pred_x, pred_y, elapsed_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True, help="Path to reference image (1000x1000)")
    parser.add_argument("--search", required=True, help="Path to search image (1000x1000)")
    parser.add_argument("--checkpoint", default="./model/checkpoints/best.pt", help="Model checkpoint path")
    parser.add_argument("--verbose", action="store_true", help="Print extra info (timing, device)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print(f"Device: {device}", file=sys.stderr)

    model = load_model(args.checkpoint, device)

    pred_x, pred_y, elapsed_ms = predict(model, args.reference, args.search, device)

    # Required output format: x,y
    print(f"{pred_x:.2f},{pred_y:.2f}")

    if args.verbose:
        print(f"Time: {elapsed_ms:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
