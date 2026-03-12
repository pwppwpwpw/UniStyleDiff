from typing import Optional

import torch
from torch import nn

from ..models.attn_processor import StyleAttnProcessor
from ..models.content_injector import ContentInjector, ConvInWithInjection
from ..models.style_injector import StyleInjector


class UniStyleDiffImageModel(nn.Module):
    """
    Stage-I model for image stylization with content/style injection on SD v1.5 UNet.
    """

    def __init__(
        self,
        unet,
        vae,
        scheduler,
        style_tokens: int = 4,
        style_dim: int = 768,
        clip_model: str = "openai/clip-vit-large-patch14",
        grayscale_ratio: float = 0.05,
        style_scale: float = 1.0,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler

        self.content_injector = ContentInjector(grayscale_ratio=grayscale_ratio)
        self.style_injector = StyleInjector(
            clip_model=clip_model,
            token_dim=style_dim,
            num_tokens=style_tokens,
            trainable_encoder=False,
        )

        self._style_processors = self._attach_style_processors(style_dim, style_scale)
        self.unet.conv_in = ConvInWithInjection(self.unet.conv_in, self.content_injector)

    def _attach_style_processors(self, style_dim: int, scale: float):
        processors = dict(self.unet.attn_processors)
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == "Attention" and name.endswith("attn2"):
                key = f"{name}.processor"
                hidden_size = module.to_q.in_features
                processors[key] = StyleAttnProcessor(hidden_size=hidden_size, cross_attention_dim=style_dim, scale=scale)
        self.unet.set_attn_processor(processors)
        return processors

    def set_style_tokens(self, tokens: Optional[torch.Tensor]) -> None:
        for processor in self._style_processors.values():
            if isinstance(processor, StyleAttnProcessor):
                processor.set_style_tokens(tokens)

    def encode_latents(self, images: torch.Tensor) -> torch.Tensor:
        images = images * 2.0 - 1.0
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        content: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if content is not None:
            self.content_injector.set_condition(content, target_hw=latents.shape[-2:])
        else:
            self.content_injector.clear()

        if style is not None:
            tokens = self.style_injector(style)
            if tokens.shape[0] != latents.shape[0]:
                tokens = tokens.repeat(latents.shape[0], 1, 1)
            self.set_style_tokens(tokens)
        else:
            self.set_style_tokens(None)

        # zero text conditioning for text-free stylization
        encoder_hidden_states = torch.zeros(latents.shape[0], 1, self.unet.config.cross_attention_dim, device=latents.device)

        noise_pred = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample
        return noise_pred
