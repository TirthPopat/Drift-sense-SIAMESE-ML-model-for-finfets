#!/usr/bin/env python3
"""CLI to generate a Drift-Sense synthetic dataset split.

Extends the original generator with three additions:

  1. Rotation: the search image can be rotated by a small random angle
     (problem statement allows ~1-2 deg). Rotation is applied to the search
     image ONLY (simulating the tool's second visit being slightly
     misaligned relative to the first), around the image's own center. The
     ground-truth (x, y) is re-derived through the same rotation matrix, so
     it stays exactly correct after rotation -- it is NOT re-measured or
     approximated.

  2. Randomized acquisition noise: with --randomize-acquisition, each
     sample draws its own dose/drift/blur/etc. from realistic ranges
     instead of every sample in a run sharing one fixed noise setting.
     This gives real diversity across noise levels within a single dataset
     split, which the validation requirements ask for explicitly.

  3. RGB mode: with --rgb, the final grayscale image is converted to a
     pseudo-RGB "optical microscope" style image (channel replication +
     independent per-channel noise + a small random color tint). This is
     a deliberately simple post-processing approximation, NOT a physical
     optical image-formation model -- documented here so it can be
     honestly described as such in the presentation. It targets the
     bonus "RGB optical-image extension" rubric item, to be attempted only
     after the core grayscale SEM solution is solid.

Example:
    python generate_dataset.py --num-samples 20 --split train \
        --architectures dram_1x finfet_10nm --output-dir ./output --seed 42 \
        --rotation-max-deg 2.0 --randomize-acquisition --rgb
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS


# --- Ranges used when --randomize-acquisition is set. Each sample draws a
# fresh value per field from U(low, high), overriding the corresponding
# GenerationParams default for that one sample only. Ranges are centered
# around the pipeline's existing defaults / evaluate.py's noise-level sweep,
# widened to cover clean -> severe acquisition conditions in one dataset.
ACQUISITION_RANGES = {
    "dose_search": (25.0, 900.0),
    "shear_amplitude_px": (0.2, 4.5),
    "drift_jitter_px": (0.1, 2.0),
    "detector_noise_sigma_search": (2.0, 12.0),
    "speckle_sigma": (0.0, 0.3),
    "salt_pepper_prob": (0.0, 0.012),
    "charging_streak_prob": (0.0, 3.5),
    "charging_streak_intensity": (0.0, 2.2),
    "vignette_strength": (0.0, 0.3),
    "gamma": (0.75, 1.35),
    "barrel_distortion_k": (-0.05, 0.05),
    "astigmatism_ratio": (0.8, 1.3),
}


def sample_randomized_params(base_params: GenerationParams, rng: np.random.Generator) -> GenerationParams:
    """Return a copy of base_params with each field in ACQUISITION_RANGES
    redrawn uniformly at random for this one sample."""
    overrides = {field: rng.uniform(lo, hi) for field, (lo, hi) in ACQUISITION_RANGES.items()}
    return GenerationParams(**{**base_params.as_dict(), **overrides})


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate img by angle_deg around its own center. Edge pixels are
    replicated rather than zero-padded, since a real re-scan wouldn't
    introduce hard black borders."""
    h, w = img.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def rotate_point(x: float, y: float, angle_deg: float, w: int, h: int) -> tuple:
    """Apply the exact same rotation used by apply_rotation() to a single
    (x, y) point, so ground truth stays consistent with the rotated image."""
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    pt = np.array([x, y, 1.0])
    new_x, new_y = M @ pt
    return float(new_x), float(new_y)


def to_pseudo_rgb(
    img_gray: np.ndarray,
    rng: np.random.Generator,
    tint_strength: float = 0.06,
    channel_noise_sigma: float = 3.0,
) -> np.ndarray:
    """Convert a flattened grayscale composite into a pseudo-RGB image, as
    a simple stand-in for an optical-microscope color capture. This is a
    deliberate approximation (channel replication + independent per-channel
    gain/noise), not a modeled optical color-formation process -- suitable
    for the bonus RGB-extension rubric item, and should be described as an
    approximation rather than a physically derived color image.
    """
    base = img_gray.astype(np.float64)
    # Small independent per-channel gain, so the tint differs from a flat
    # gray copy -- e.g. a slight green/blue cast typical of some optical
    # inspection setups -- but stays close to the original brightness.
    gains = 1.0 + rng.uniform(-tint_strength, tint_strength, size=3)
    channels = []
    for gain in gains:
        ch = base * gain
        # Independent sensor noise per channel -- these are three separate
        # detector wells even in a single-shot color camera, so noise must
        # not be copy-pasted across channels.
        ch = ch + rng.normal(0, channel_noise_sigma, size=base.shape)
        channels.append(np.clip(ch, 0, 255).astype(np.uint8))
    return cv2.merge(channels)  # BGR order for cv2.imwrite


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-spot-size-nm", type=float, default=GenerationParams.beam_spot_size_nm)
    p.add_argument("--collapse-threshold-nm", type=float, default=GenerationParams.collapse_threshold_nm)
    p.add_argument("--dose-reference", type=float, default=GenerationParams.dose_reference)
    p.add_argument("--dose-search", type=float, default=GenerationParams.dose_search)
    p.add_argument("--shear-amplitude-px", type=float, default=GenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=GenerationParams.drift_jitter_px)
    p.add_argument("--astigmatism-ratio", type=float, default=GenerationParams.astigmatism_ratio)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--gamma", type=float, default=GenerationParams.gamma)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=GenerationParams.charging_streak_prob)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--linewidth-bias-nm", type=float, default=GenerationParams.linewidth_bias_nm)
    p.add_argument("--corner-rounding-px", type=float, default=GenerationParams.corner_rounding_px)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)

    # --- New options ---
    p.add_argument(
        "--rotation-max-deg", type=float, default=0.0,
        help="Max absolute rotation (deg) applied to the search image only. "
             "Each sample draws angle ~ U(-max, +max). Problem statement allows ~1-2 deg. Default 0 = off.",
    )
    p.add_argument(
        "--randomize-acquisition", action="store_true",
        help="Draw each sample's noise/drift/blur settings independently from "
             "ACQUISITION_RANGES instead of using one fixed setting for the whole run. "
             "Produces a dataset that spans clean -> severe conditions.",
    )
    p.add_argument(
        "--rgb", action="store_true",
        help="Save reference/search images as pseudo-RGB (3-channel) instead of grayscale. "
             "Approximation for the bonus optical-image extension -- see to_pseudo_rgb().",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    base_params = GenerationParams(
        beam_spot_size_nm=args.beam_spot_size_nm,
        collapse_threshold_nm=args.collapse_threshold_nm,
        dose_reference=args.dose_reference,
        dose_search=args.dose_search,
        shear_amplitude_px=args.shear_amplitude_px,
        drift_jitter_px=args.drift_jitter_px,
        astigmatism_ratio=args.astigmatism_ratio,
        vignette_strength=args.vignette_strength,
        gamma=args.gamma,
        barrel_distortion_k=args.barrel_distortion_k,
        charging_streak_prob=args.charging_streak_prob,
        charging_streak_intensity=args.charging_streak_intensity,
        speckle_sigma=args.speckle_sigma,
        salt_pepper_prob=args.salt_pepper_prob,
        linewidth_bias_nm=args.linewidth_bias_nm,
        corner_rounding_px=args.corner_rounding_px,
        mat_size_nm=args.mat_size_nm,
        strip_width_nm=args.strip_width_nm,
        boundary_bias=args.boundary_bias,
    )

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture",
        "rotation_deg", "is_rgb", "randomized_acquisition",
        "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
        "dose_search", "shear_amplitude_px", "drift_jitter_px",
        "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
        "charging_streak_prob", "charging_streak_intensity",
        "speckle_sigma", "salt_pepper_prob",
        "linewidth_bias_nm", "corner_rounding_px",
        "mat_size_nm", "strip_width_nm", "boundary_bias", "seed",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]

            # Per-sample noise diversity, if requested.
            sample_params = (
                sample_randomized_params(base_params, rng) if args.randomize_acquisition else base_params
            )

            sample = generate_sample(architecture, rng, sample_params)
            reference_img = sample["reference_img"]
            search_img = sample["search_img"]
            gt_x, gt_y = sample["gt_x"], sample["gt_y"]

            # Rotation: search image only, ground truth re-derived exactly
            # via the same rotation matrix (not re-measured/approximated).
            angle_deg = 0.0
            if args.rotation_max_deg > 0:
                angle_deg = float(rng.uniform(-args.rotation_max_deg, args.rotation_max_deg))
                h, w = search_img.shape[:2]
                search_img = apply_rotation(search_img, angle_deg)
                gt_x, gt_y = rotate_point(gt_x, gt_y, angle_deg, w, h)

            # RGB conversion, if requested. Independent noise per image (and
            # per channel), matching the "don't reuse noise" requirement.
            if args.rgb:
                reference_img = to_pseudo_rgb(reference_img, rng)
                search_img = to_pseudo_rgb(search_img, rng)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, reference_img)
            cv2.imwrite(search_path, search_img)

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i,
                "reference_path": ref_path,
                "search_path": search_path,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture,
                "rotation_deg": angle_deg,
                "is_rgb": args.rgb,
                "randomized_acquisition": args.randomize_acquisition,
                **sample_params.as_dict(),
                "seed": args.seed,
            })
            print(
                f"[{i + 1}/{args.num_samples}] {architecture} -> gt=({gt_x:.1f}, {gt_y:.1f}) "
                f"rot={angle_deg:+.2f}deg rgb={args.rgb}"
            )

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()