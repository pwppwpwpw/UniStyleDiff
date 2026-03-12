from typing import List, Optional

import torch
from torch import nn

from ..models.icm import ICM
from ..pipelines.pipeline_image import UniStyleDiffImageModel


class ICMController:
    def __init__(self, unet: nn.Module, icm: ICM, target_module_keys: Optional[List[str]] = None):
        self.unet = unet
        self.icm = icm
        self.target_module_keys = target_module_keys or ["attentions"]
        self.num_frames = 1
        self.collect_attn = False
        self._hooks = []

    def attach(self) -> None:
        for name, module in self.unet.named_modules():
            if not self.target_module_keys or any(key in name for key in self.target_module_keys):
                if hasattr(module, "forward"):
                    hook = module.register_forward_hook(self._make_hook())
                    self._hooks.append(hook)

    def _make_hook(self):
        def hook(_module, _inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4 and self.num_frames > 1:
                return self.icm(out, num_frames=self.num_frames, return_attn=self.collect_attn)
            return out

        return hook

    def clear(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []


class UniStyleDiffVideoModel(nn.Module):
    """
    Stage-II model: image model + ICM for temporal consistency.
    """

    def __init__(
        self,
        image_model: UniStyleDiffImageModel,
        icm_dim: int = 320,
        icm_heads: int = 8,
        icm_layers: int = 2,
        icm_targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.image_model = image_model
        self.icm = ICM(dim=icm_dim, num_heads=icm_heads, num_layers=icm_layers)
        self.icm_controller = ICMController(self.image_model.unet, self.icm, target_module_keys=icm_targets)
        self.icm_controller.attach()

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_frames: int,
        content: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
        collect_attn: bool = False,
    ) -> torch.Tensor:
        self.icm_controller.num_frames = num_frames
        self.icm_controller.collect_attn = collect_attn
        return self.image_model(latents, timesteps, content=content, style=style)
