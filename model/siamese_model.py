"""
Siamese Correlation Network for Drift-Sense localization.

Architecture:
  1. Shared ResNet-34 encoder (modified for 1-channel grayscale)
     - Takes features through layer3 (stride=16, 256 channels)
     - 1x1 conv projects to 128 channels
  2. Cross-correlation between template and search feature maps
  3. Soft-argmax for differentiable sub-pixel (x, y) prediction

The reference image (1000x1000 @ 1nm/px) is downsampled 10x to 100x100
so it matches the search image's scale (1000x1000 @ 10nm/px). Both go
through the same encoder, then cross-correlation finds the match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseCorrelationNet(nn.Module):
    """Siamese network with cross-correlation matching head.

    Encodes template and search images with a shared ResNet-18 backbone
    (up to layer2, stride=8), then cross-correlates the feature maps to
    produce a match heatmap. A differentiable soft-argmax extracts the
    sub-pixel peak location, which is mapped back to search-image pixel
    coordinates.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Load ResNet-18 and modify for 1-channel grayscale input
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # Replace first conv: 3-channel -> 1-channel
        # Average the pretrained weights across the 3 input channels so
        # the initial features are still meaningful on grayscale.
        old_conv = resnet.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # stride 1, 64 channels
        self.layer2 = resnet.layer2   # stride 2, 128 channels
        # Total stride from input: 2 (conv1) * 2 (maxpool) * 2 (layer2) = 8

        self.feat_channels = 128
        self.total_stride = 8

        # Batch-norm on correlation map for stable training
        self.corr_bn = nn.BatchNorm2d(1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: grayscale image -> feature map at stride 8."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def cross_correlate(
        self, search_feat: torch.Tensor, template_feat: torch.Tensor
    ) -> torch.Tensor:
        """Depth-wise cross-correlation between search and template features.

        For each sample in the batch, the template feature map is used as
        a convolution kernel slid over the search feature map. The result
        is a 2D score map where higher values indicate better matches.
        """
        B = search_feat.size(0)
        corr_maps = []
        for i in range(B):
            # template as conv kernel: (1, C, th, tw)
            kernel = template_feat[i].unsqueeze(0)
            # search feature: (1, C, sh, sw)
            search = search_feat[i].unsqueeze(0)
            # Cross-correlation via conv2d (valid mode)
            corr = F.conv2d(search, kernel)  # (1, 1, H_out, W_out)
            corr_maps.append(corr)
        corr_map = torch.cat(corr_maps, dim=0)  # (B, 1, H_out, W_out)
        corr_map = self.corr_bn(corr_map)
        return corr_map

    def soft_argmax(self, heatmap: torch.Tensor) -> tuple:
        """Differentiable soft-argmax for training.
        
        Applies softmax over the flattened heatmap to get a probability
        distribution, then computes the expected x and y coordinates.
        This provides smooth gradients during training. The hybrid pipeline
        will handle local peak-picking during inference.
        """
        B, _, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, -1)  # (B, H*W)

        # Temperature-scaled softmax for sharper peak
        softmax_weights = F.softmax(heatmap_flat * 10.0, dim=1)  # (B, H*W)

        # Coordinate grids
        device = heatmap.device
        pos_y = torch.arange(H, dtype=torch.float32, device=device)
        pos_x = torch.arange(W, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        grid_x_flat = grid_x.reshape(-1).unsqueeze(0)  # (1, H*W)
        grid_y_flat = grid_y.reshape(-1).unsqueeze(0)

        # Expected coordinates (sub-pixel)
        pred_x = (softmax_weights * grid_x_flat).sum(dim=1)  # (B,)
        pred_y = (softmax_weights * grid_y_flat).sum(dim=1)

        return pred_x, pred_y

    def corr_to_pixel(
        self, corr_x: torch.Tensor, corr_y: torch.Tensor,
        template_pixel_size: int = 100
    ) -> tuple:
        """Convert correlation-map coordinates to search-image pixel coordinates.

        Correlation position (cx, cy) means the template's top-left was
        placed at feature position (cx, cy). The center of the matched
        region in pixel space is:
            pixel_x = cx * stride + template_pixel_size / 2
            pixel_y = cy * stride + template_pixel_size / 2
        """
        half_template = template_pixel_size / 2.0
        pixel_x = corr_x * self.total_stride + half_template
        pixel_y = corr_y * self.total_stride + half_template
        return pixel_x, pixel_y

    def forward(
        self, template: torch.Tensor, search: torch.Tensor,
        template_pixel_size: int = 100
    ) -> dict:
        """
        Args:
            template: (B, 1, 100, 100) — downsampled reference
            search:   (B, 1, 1000, 1000) — search image
            template_pixel_size: size of template in search-image pixels

        Returns:
            dict with 'pred_x', 'pred_y' (pixel coords), 'heatmap'
        """
        template_feat = self.encode(template)   # (B, 128, 13, 13)
        search_feat = self.encode(search)       # (B, 128, 125, 125)

        corr_map = self.cross_correlate(search_feat, template_feat)  # (B, 1, H, W)

        corr_x, corr_y = self.soft_argmax(corr_map)
        pred_x, pred_y = self.corr_to_pixel(corr_x, corr_y, template_pixel_size)

        return {
            "pred_x": pred_x,
            "pred_y": pred_y,
            "heatmap": corr_map,
            "corr_x": corr_x,
            "corr_y": corr_y,
        }


def pixel_to_corr(gt_x, gt_y, stride=8, template_pixel_size=100):
    """Convert ground-truth pixel coordinates to correlation-map coordinates.

    Used to create ground-truth targets for heatmap supervision.
    """
    half = template_pixel_size / 2.0
    corr_x = (gt_x - half) / stride
    corr_y = (gt_y - half) / stride
    return corr_x, corr_y


def make_gt_heatmap(gt_corr_x, gt_corr_y, map_h, map_w, sigma=2.0, device="cpu"):
    """Create a ground-truth Gaussian heatmap centered at (gt_corr_x, gt_corr_y).

    Used as target for heatmap-level supervision alongside coordinate loss.
    """
    B = gt_corr_x.shape[0]
    pos_y = torch.arange(map_h, dtype=torch.float32, device=device)
    pos_x = torch.arange(map_w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    cx = gt_corr_x.view(B, 1, 1)
    cy = gt_corr_y.view(B, 1, 1)

    heatmap = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.unsqueeze(1)  # (B, 1, H, W)
