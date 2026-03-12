from typing import Optional

import torch
from torch import nn


class StyleEncoder(nn.Module):
    """
    CLIP image encoder wrapper. Keeps encoder frozen by default.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", trainable: bool = False):
        super().__init__()
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise ImportError("transformers is required for StyleEncoder") from exc

        self.encoder = CLIPVisionModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images in [0,1], resize + normalize to CLIP expectations
        pixel_values = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean) / std
        outputs = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        return outputs.pooler_output


class StyleProjector(nn.Module):
    def __init__(self, in_dim: int, token_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.proj = nn.Linear(in_dim, num_tokens * token_dim)

    def forward(self, style_vec: torch.Tensor) -> torch.Tensor:
        b = style_vec.shape[0]
        tokens = self.proj(style_vec).view(b, self.num_tokens, self.token_dim)
        return tokens


class StyleInjector(nn.Module):
    """
    Extract style vector and produce style tokens for cross-attention.
    """

    def __init__(
        self,
        clip_model: str = "openai/clip-vit-large-patch14",
        token_dim: int = 768,
        num_tokens: int = 4,
        trainable_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = StyleEncoder(model_name=clip_model, trainable=trainable_encoder)
        self.projector = StyleProjector(in_dim=self.encoder.encoder.config.hidden_size, token_dim=token_dim, num_tokens=num_tokens)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        style_vec = self.encoder(images)
        return self.projector(style_vec)
