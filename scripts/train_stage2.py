import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from unistylediff.data.datasets import VideoStyleDataset
from unistylediff.pipelines.pipeline_image import UniStyleDiffImageModel
from unistylediff.pipelines.pipeline_video import UniStyleDiffVideoModel
from unistylediff.utils.config import load_config, get, save_config
from unistylediff.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(get(cfg, "seed", default=42))
    device = torch.device(get(cfg, "device", default="cuda") if torch.cuda.is_available() else "cpu")

    output_dir = Path(get(cfg, "train", "output_dir", default="./runs/unistylediff_stage2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, str(output_dir / "config.json"))

    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    except ImportError as exc:
        raise ImportError("diffusers is required for training") from exc

    model_id = get(cfg, "model", "pretrained", default="runwayml/stable-diffusion-v1-5")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    for p in vae.parameters():
        p.requires_grad = False

    stage1 = torch.load(args.stage1_ckpt, map_location="cpu")
    unet.load_state_dict(stage1["unet"], strict=False)

    image_model = UniStyleDiffImageModel(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        style_tokens=get(cfg, "style", "num_tokens", default=4),
        style_dim=get(cfg, "style", "token_dim", default=768),
        clip_model=get(cfg, "style", "clip_model", default="openai/clip-vit-large-patch14"),
        grayscale_ratio=get(cfg, "content", "grayscale_ratio", default=0.05),
        style_scale=get(cfg, "style", "scale", default=1.0),
    ).to(device)

    # Freeze SD + content/style injection
    for p in image_model.parameters():
        p.requires_grad = False

    video_model = UniStyleDiffVideoModel(
        image_model=image_model,
        icm_dim=get(cfg, "icm", "dim", default=320),
        icm_heads=get(cfg, "icm", "heads", default=8),
        icm_layers=get(cfg, "icm", "layers", default=2),
        icm_targets=get(cfg, "icm", "targets", default=[]),
    ).to(device)

    for p in video_model.icm.parameters():
        p.requires_grad = True

    dataset = VideoStyleDataset(
        video_root=get(cfg, "data", "video_dir"),
        style_dir=get(cfg, "data", "style_dir"),
        num_frames=get(cfg, "data", "num_frames", default=16),
        frame_stride=get(cfg, "data", "frame_stride", default=4),
        image_size=get(cfg, "data", "image_size", default=512),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=get(cfg, "train", "num_workers", default=4))

    optim = torch.optim.AdamW(video_model.icm.parameters(), lr=get(cfg, "train", "lr", default=1e-5))

    video_model.train()
    for epoch in range(get(cfg, "train", "epochs", default=50)):
        for step, (video, style) in enumerate(loader):
            # video: [F,3,H,W]
            video = video.to(device)
            style = style.to(device)

            b, f, c, h, w = 1, video.shape[0], video.shape[1], video.shape[2], video.shape[3]
            video_flat = video.view(f, c, h, w)
            latents = image_model.encode_latents(video_flat)
            latents = latents.view(b * f, latents.shape[1], latents.shape[2], latents.shape[3])

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            noise_pred = video_model(noisy_latents, timesteps, num_frames=f, content=video_flat, style=style)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (step + 1) % get(cfg, "train", "log_every", default=10) == 0:
                print(f"epoch {epoch} step {step+1}: loss={loss.item():.4f}")

        ckpt_path = output_dir / f"stage2_epoch_{epoch+1}.pt"
        torch.save({"icm": video_model.icm.state_dict()}, ckpt_path)


if __name__ == "__main__":
    main()
