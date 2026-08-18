#!/usr/bin/env python3
"""Draw the ground-truth box on a generated sample's search image, next to
its reference thumbnail, and save a side-by-side comparison PNG.

Handles both grayscale and RGB samples (reads whichever the PNG actually
is, rather than forcing grayscale), and if the sample was rotated
(rotation_deg column present and non-zero in the manifest), draws the
ground-truth box rotated to match -- using the exact same rotation matrix
generate_dataset.py used -- instead of a stale axis-aligned rectangle.
A crosshair is also drawn at the precise (gt_x, gt_y) center, since that
point is always exactly correct regardless of box drawing.

Example:
    python visualize_sample.py --output-dir ./output --split train --id 0
"""

import argparse
import csv
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--split", default="train")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--save-path", default=None)
    return p.parse_args()


def read_image_auto(path: str) -> np.ndarray:
    """Read a PNG as-is (grayscale stays 1-channel, RGB stays 3-channel),
    then return it as a 3-channel BGR image for consistent drawing/stacking."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def rotate_box_corners(x0: float, y0: float, w: float, h: float, angle_deg: float, img_w: int, img_h: int):
    """Return the 4 corners of the (originally axis-aligned) gt box, rotated
    by angle_deg around the image center -- the same transform applied to
    the search image itself in generate_dataset.py's apply_rotation()."""
    corners = np.array([
        [x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h],
    ], dtype=np.float64)
    center = ((img_w - 1) / 2.0, (img_h - 1) / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    ones = np.ones((4, 1))
    corners_h = np.hstack([corners, ones])
    rotated = corners_h @ M.T
    return rotated.astype(np.int32)


def main():
    args = parse_args()
    split_dir = os.path.join(args.output_dir, args.split)
    manifest_path = os.path.join(split_dir, "manifest.csv")

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if int(r["id"]) == args.id), None)
    if row is None:
        raise ValueError(f"id {args.id} not found in {manifest_path}")

    reference = read_image_auto(row["reference_path"])
    search = read_image_auto(row["search_path"])

    search_annotated = search.copy()
    x0, y0, w, h = (float(row["gt_box_x"]), float(row["gt_box_y"]), float(row["gt_box_w"]), float(row["gt_box_h"]))

    # rotation_deg / is_rgb are new columns -- default to 0/False for
    # manifests generated before this change, so old datasets still work.
    angle_deg = float(row.get("rotation_deg", 0.0) or 0.0)
    img_h, img_w = search_annotated.shape[:2]

    if abs(angle_deg) > 1e-6:
        box_pts = rotate_box_corners(x0, y0, w, h, angle_deg, img_w, img_h)
        cv2.polylines(search_annotated, [box_pts], isClosed=True, color=(0, 0, 255), thickness=2)
    else:
        cv2.rectangle(
            search_annotated,
            (int(round(x0)), int(round(y0))),
            (int(round(x0 + w)), int(round(y0 + h))),
            (0, 0, 255),
            2,
        )

    # Crosshair at the exact ground-truth center -- always correct,
    # independent of whether the box itself is a good visual fit.
    gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
    cx, cy = int(round(gt_x)), int(round(gt_y))
    cv2.drawMarker(search_annotated, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)

    ref_resized = cv2.resize(
        reference,
        (search_annotated.shape[1], search_annotated.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    combined = np.hstack([ref_resized, search_annotated])

    save_path = args.save_path or os.path.join(split_dir, f"visualize_{args.id:05d}.png")
    cv2.imwrite(save_path, combined)
    print(f"architecture={row['architecture']}  gt=({row['gt_x']}, {row['gt_y']})  rotation_deg={angle_deg:.2f}")
    print(f"Saved comparison to {save_path}")


if __name__ == "__main__":
    main()