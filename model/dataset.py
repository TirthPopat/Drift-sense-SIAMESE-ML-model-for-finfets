"""
PyTorch Dataset for Drift-Sense image pairs.

Reads manifest.csv, loads reference + search PNGs, downsamples the
reference by 10x (to match search-image scale), normalizes to [0, 1],
and returns (template, search, gt_x, gt_y).
"""

import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DriftSenseDataset(Dataset):
    """Load reference/search pairs from a generated dataset split.

    The reference (1000x1000 @ 1nm/px) is downsampled 10x to 100x100 so
    it matches the search image's pixel scale. Both are returned as
    single-channel float32 tensors normalized to [0, 1].
    """

    DOWNSAMPLE_FACTOR = 10

    def __init__(
        self,
        manifest_path: str,
        max_samples: int | None = None,
        augment: bool = False,
    ):
        self.manifest_path = manifest_path
        self.split_dir = os.path.dirname(manifest_path)
        self.augment = augment

        with open(manifest_path) as f:
            self.rows = list(csv.DictReader(f))

        if max_samples is not None:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # Load images
        ref_path = row["reference_path"]
        search_path = row["search_path"]

        # Handle relative paths (manifest may store relative to project root)
        if not os.path.isabs(ref_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if ref_path.startswith("./") or ref_path.startswith(".\\"):
                ref_path = ref_path[2:]
            if search_path.startswith("./") or search_path.startswith(".\\"):
                search_path = search_path[2:]
            ref_path = os.path.join(project_root, ref_path)
            search_path = os.path.join(project_root, search_path)

        reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

        if reference is None:
            raise FileNotFoundError(f"Cannot read reference: {ref_path}")
        if search is None:
            raise FileNotFoundError(f"Cannot read search: {search_path}")

        # Downsample reference 10x to match search-image scale
        h, w = reference.shape
        template = cv2.resize(
            reference,
            (w // self.DOWNSAMPLE_FACTOR, h // self.DOWNSAMPLE_FACTOR),
            interpolation=cv2.INTER_AREA,
        )

        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])

        # Augmentations for noise robustness
        if self.augment:
            # 1. Geometric flips
            if np.random.random() > 0.5:
                template = np.fliplr(template).copy()
                search = np.fliplr(search).copy()
                gt_x = search.shape[1] - gt_x

            if np.random.random() > 0.5:
                template = np.flipud(template).copy()
                search = np.flipud(search).copy()
                gt_y = search.shape[0] - gt_y

            # 2. Random contrast / brightness jitter
            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.7, 1.3)  # contrast
                beta = np.random.uniform(-20, 20)    # brightness
                search = np.clip(alpha * search + beta, 0, 255).astype(np.uint8)

            # 3. Random Gaussian noise injection to search image
            if np.random.random() > 0.5:
                noise_sigma = np.random.uniform(2.0, 15.0)
                noise = np.random.normal(0, noise_sigma, search.shape)
                search = np.clip(search.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 4. Random slight blur
            if np.random.random() > 0.4:
                search = cv2.GaussianBlur(search, (3, 3), 0)

        # Normalize to [0, 1] and add channel dimension
        template_t = torch.from_numpy(template.astype(np.float32) / 255.0).unsqueeze(0)
        search_t = torch.from_numpy(search.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "template": template_t,    # (1, 100, 100)
            "search": search_t,        # (1, 1000, 1000)
            "gt_x": torch.tensor(gt_x, dtype=torch.float32),
            "gt_y": torch.tensor(gt_y, dtype=torch.float32),
            "id": int(row["id"]),
            "architecture": row["architecture"],
        }


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 4,
    num_workers: int = 2,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
):
    """Create train and val dataloaders from separate manifest files."""
    train_ds = DriftSenseDataset(
        train_manifest, max_samples=max_train_samples, augment=True
    )
    val_ds = DriftSenseDataset(
        val_manifest, max_samples=max_val_samples, augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
