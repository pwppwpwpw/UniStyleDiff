from typing import List, Optional, Tuple

import torch
from torch import nn


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [N, F, C]
        x_in = self.proj_in(x)
        out, attn = self.attn(x_in, x_in, x_in, need_weights=return_attn, average_attn_weights=False)
        out = self.proj_out(out)
        out = out + x
        return out, attn


class ICM(nn.Module):
    """
    Inter-frame Consistency Module with stacked temporal self-attention blocks.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([TemporalSelfAttention(dim, num_heads, dropout) for _ in range(num_layers)])
        self.last_attn: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, num_frames: int, return_attn: bool = False) -> torch.Tensor:
        # x: [B*F, C, H, W] -> reshape to [B*H*W, F, C]
        b_f, c, h, w = x.shape
        assert b_f % num_frames == 0, "num_frames does not divide batch"
        b = b_f // num_frames

        x = x.view(b, num_frames, c, h, w).permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(b * h * w, num_frames, c)

        self.last_attn = []
        for layer in self.layers:
            x, attn = layer(x, return_attn=return_attn)
            if return_attn and attn is not None:
                self.last_attn.append(attn.detach())

        x = x.view(b, h, w, num_frames, c).permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(b_f, c, h, w)
        return x
