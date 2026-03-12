from typing import List

import torch


class MDPSampler:
    """
    Motion-Dynamics Preserved sampling: uses temporal loss from ICM attention maps.
    """

    def __init__(self, video_model, scheduler, guidance_scale: float = 1.0, time_scale: float = 1.0):
        self.video_model = video_model
        self.scheduler = scheduler
        self.guidance_scale = guidance_scale
        self.time_scale = time_scale

    def temporal_loss(self, attn_ref: List[torch.Tensor], attn_gen: List[torch.Tensor]) -> torch.Tensor:
        loss = 0.0
        for ref, gen in zip(attn_ref, attn_gen):
            loss = loss + (ref - gen).abs().mean()
        return loss

    def step(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        num_frames: int,
        content: torch.Tensor,
        style: torch.Tensor,
        reference_latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = latents.detach().requires_grad_(True)

        # Reference branch attention maps
        _ = self.video_model(
            reference_latents,
            t,
            num_frames=num_frames,
            content=content,
            style=None,
            collect_attn=True,
        )
        attn_ref = self.video_model.icm.last_attn

        # Generated branch attention maps
        noise_pred = self.video_model(
            latents,
            t,
            num_frames=num_frames,
            content=content,
            style=style,
            collect_attn=True,
        )
        attn_gen = self.video_model.icm.last_attn

        loss_temp = self.temporal_loss(attn_ref, attn_gen)
        grad = torch.autograd.grad(loss_temp, latents, retain_graph=False, create_graph=False)[0]

        # Standard DDIM update with temporal guidance
        prev_sample = self.scheduler.step(noise_pred, t, latents).prev_sample
        guided = prev_sample - self.guidance_scale * self.time_scale * grad
        return guided.detach()
