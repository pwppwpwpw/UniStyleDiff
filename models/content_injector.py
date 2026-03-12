from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .content_fusion import ContentFusionModule, resize_condition


class ContentInjector(nn.Module):
    """
    Build fused content condition and inject into UNet conv_in features:
        Z'_t = lambda * F_fusion + ConvIn(Z_t)
    """

    def __init__(self, grayscale_ratio: float = 0.05):
        super().__init__()
        self.fusion = ContentFusionModule()
        self.grayscale_ratio = grayscale_ratio
        self.lambda_param = nn.Parameter(torch.tensor(1.0))
        self._cached_fusion: Optional[torch.Tensor] = None

    @staticmethod
    def _to_gray(x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W] -> [B,1,H,W]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    @staticmethod
    def _normalize_map(x: torch.Tensor) -> torch.Tensor:
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def build_condition(
        self,
        content: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
        seg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # content: [B,3,H,W]
        gray = self._to_gray(content)
        if edge is None:
            # Fallback: simple Sobel edge if no external detector provided.
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=content.device, dtype=content.dtype).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=content.device, dtype=content.dtype).view(1, 1, 3, 3)
            edge_x = F.conv2d(gray, sobel_x, padding=1)
            edge_y = F.conv2d(gray, sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        if seg is None:
            # Fallback: use zeros when segmentation map is absent.
            seg = torch.zeros_like(gray)

        edge = self._normalize_map(edge)
        seg = self._normalize_map(seg)
        gray = self._normalize_map(gray)

        mixed_edge = edge * (1.0 - self.grayscale_ratio) + gray * self.grayscale_ratio
        # Build 3-channel condition: [edge+gray, seg, gray]
        cond = torch.cat([mixed_edge, seg, gray], dim=1)
        return cond

    def set_condition(
        self,
        content: torch.Tensor,
        edge: Optional[torch.Tensor] = None,
        seg: Optional[torch.Tensor] = None,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> None:
        cond = self.build_condition(content, edge=edge, seg=seg)
        if target_hw is not None:
            cond = resize_condition(cond, target_hw)
        self._cached_fusion = self.fusion(cond)

    def inject(self, conv_in_feat: torch.Tensor) -> torch.Tensor:
        if self._cached_fusion is None:
            return conv_in_feat
        fusion = resize_condition(self._cached_fusion, conv_in_feat.shape[-2:])
        return conv_in_feat + self.lambda_param * fusion

    def clear(self) -> None:
        self._cached_fusion = None


class ConvInWithInjection(nn.Module):
    def __init__(self, conv_in: nn.Module, injector: ContentInjector):
        super().__init__()
        self.conv_in = conv_in
        self.injector = injector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        return self.injector.inject(h)
