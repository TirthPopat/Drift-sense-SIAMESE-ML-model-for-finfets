#!/usr/bin/env python3
"""
Hybrid Coarse-to-Fine Localization for Drift-Sense.

Strategy (designed for <5 pixel accuracy):
  Step 1 (COARSE): Run the fast Siamese DL model on the full 1000x1000 image.
         It reliably finds the correct general area within ~15-20 pixels.
  Step 2 (FINE):   Crop a small window around the DL prediction and run
         ZNCC template matching ONLY within that tiny crop. ZNCC gives
         sub-pixel precision, and because the search window is tiny,
         it can't get lost on the wrong repeating period.

Result: DL global awareness + ZNCC sub-pixel precision = <5px accuracy.

Example:
    python model/hybrid_localize.py --reference ref.png --search search.png
    # Output: 450.32,550.17
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


def coarse_predict(model, template_t, search_t, device):
    """Step 1: DL model gives coarse (x, y) prediction."""
    with torch.no_grad():
        output = model(template_t, search_t)
    return output["pred_x"].item(), output["pred_y"].item()


def preprocess_for_matching(img):
    """Enhance structural edges and strip high-frequency speckle noise."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    filtered = cv2.GaussianBlur(enhanced, (3, 3), 0.8)
    return filtered


def fine_refine_zncc_rotation(reference_gray, search_gray, coarse_x, coarse_y,
                              window_radius=80):
    """Step 2: Enhanced ZNCC with CLAHE + Denoising + fine angular + multi-scale search.
    
    Improvements for high noise robustness:
      1. CLAHE contrast equalization (recovers faint fins in low-dose images)
      2. Gaussian smoothing to strip high-frequency speckle noise
      3. Fine angular sweep (-2.0 to +2.0 deg in 0.2 deg steps)
      4. Multi-scale search (9.8x, 10.0x, 10.2x)
      5. Parabolic sub-pixel peak interpolation
    """
    h_search, w_search = search_gray.shape[:2]
    h_ref, w_ref = reference_gray.shape[:2]

    # Preprocess both images for noise invariance
    ref_clean = preprocess_for_matching(reference_gray)
    search_clean = preprocess_for_matching(search_gray)

    # Multi-scale + fine angular grid search
    scales = [9.8, 10.0, 10.2]
    angles = np.linspace(-2.0, 2.0, 21)  # 21 angles from -2.0 to +2.0 (step 0.2 deg)
    
    best_score = -1.0
    best_x, best_y = coarse_x, coarse_y
    best_result = None
    best_crop_origin = (0, 0)
    best_tw, best_th = 0, 0

    for scale in scales:
        new_w = max(1, int(round(w_ref / scale)))
        new_h = max(1, int(round(h_ref / scale)))
        template_base = cv2.resize(ref_clean, (new_w, new_h), interpolation=cv2.INTER_AREA)
        th, tw = template_base.shape[:2]

        for angle in angles:
            # Rotate template
            M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), angle, 1.0)
            template = cv2.warpAffine(template_base, M, (tw, th), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)

            # Define search window around coarse prediction
            tl_x = int(round(coarse_x - tw / 2.0))
            tl_y = int(round(coarse_y - th / 2.0))
            
            crop_x1 = max(0, tl_x - window_radius)
            crop_y1 = max(0, tl_y - window_radius)
            crop_x2 = min(w_search, tl_x + tw + window_radius)
            crop_y2 = min(h_search, tl_y + th + window_radius)
            
            if (crop_x2 - crop_x1) <= tw or (crop_y2 - crop_y1) <= th:
                continue
                
            crop = search_clean[crop_y1:crop_y2, crop_x1:crop_x2]

            # Run ZNCC
            result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_result = result
                best_crop_origin = (crop_x1, crop_y1)
                best_tw, best_th = tw, th
                match_tl_x = crop_x1 + max_loc[0]
                match_tl_y = crop_y1 + max_loc[1]
                best_x = match_tl_x + tw / 2.0
                best_y = match_tl_y + th / 2.0

    # Sub-pixel refinement via parabolic fitting around the best ZNCC peak
    if best_result is not None:
        rh, rw = best_result.shape
        px = int(round(best_x - best_crop_origin[0] - best_tw / 2.0))
        py = int(round(best_y - best_crop_origin[1] - best_th / 2.0))
        
        # Parabolic fit in X
        if 1 <= px <= rw - 2:
            left  = float(best_result[py, px - 1])
            center = float(best_result[py, px])
            right = float(best_result[py, px + 1])
            denom = 2.0 * (2.0 * center - left - right)
            if abs(denom) > 1e-7:
                sub_x = (left - right) / denom
                best_x += sub_x
        
        # Parabolic fit in Y
        if 1 <= py <= rh - 2:
            top    = float(best_result[py - 1, px])
            center = float(best_result[py, px])
            bottom = float(best_result[py + 1, px])
            denom = 2.0 * (2.0 * center - top - bottom)
            if abs(denom) > 1e-7:
                sub_y = (top - bottom) / denom
                best_y += sub_y

    return float(best_x), float(best_y)


def hybrid_predict(model, reference_path, search_path, device,
                   window_radius=40):
    """Full hybrid pipeline: DL coarse -> ZNCC fine.

    Returns (pred_x, pred_y, elapsed_ms, coarse_x, coarse_y).
    """
    # Load images
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Cannot read reference: {reference_path}")
    if search is None:
        raise FileNotFoundError(f"Cannot read search: {search_path}")

    t0 = time.time()

    # Step 1: Coarse DL prediction
    h, w = reference.shape
    template = cv2.resize(reference, (w // 10, h // 10),
                          interpolation=cv2.INTER_AREA)
    template_t = torch.from_numpy(template.astype(np.float32) / 255.0)
    template_t = template_t.unsqueeze(0).unsqueeze(0).to(device)
    search_t = torch.from_numpy(search.astype(np.float32) / 255.0)
    search_t = search_t.unsqueeze(0).unsqueeze(0).to(device)

    coarse_x, coarse_y = coarse_predict(model, template_t, search_t, device)

    # Clamp coarse prediction to valid range
    coarse_x = max(50, min(coarse_x, search.shape[1] - 50))
    coarse_y = max(50, min(coarse_y, search.shape[0] - 50))

    # Step 2: Fine Rotated ZNCC refinement
    refined_x, refined_y = fine_refine_zncc_rotation(
        reference, search, coarse_x, coarse_y,
        window_radius=window_radius,
    )

    # Clamp final result
    refined_x = max(0, min(refined_x, search.shape[1]))
    refined_y = max(0, min(refined_y, search.shape[0]))

    elapsed_ms = (time.time() - t0) * 1000

    return refined_x, refined_y, elapsed_ms, coarse_x, coarse_y


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--reference", required=True,
                        help="Path to reference image (1000x1000)")
    parser.add_argument("--search", required=True,
                        help="Path to search image (1000x1000)")
    parser.add_argument("--checkpoint",
                        default="./model/checkpoints/best.pt",
                        help="Model checkpoint path")
    parser.add_argument("--window-radius", type=int, default=40,
                        help="ZNCC refinement window half-size (px)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extra info")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    pred_x, pred_y, elapsed_ms, cx, cy = hybrid_predict(
        model, args.reference, args.search, device,
        window_radius=args.window_radius,
    )

    # Required output format: x,y
    print(f"{pred_x:.2f},{pred_y:.2f}")

    if args.verbose:
        print(f"Coarse DL: ({cx:.1f}, {cy:.1f})", file=sys.stderr)
        print(f"Refined:   ({pred_x:.2f}, {pred_y:.2f})", file=sys.stderr)
        print(f"Time: {elapsed_ms:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
