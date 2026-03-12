import argparse
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from unistylediff.data.datasets import ImageStyleDataset
from unistylediff.pipelines.pipeline_image import UniStyleDiffImageModel
from unistylediff.utils.config import load_config, get, save_config
from unistylediff.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(get(cfg, "seed", default=42))
    device = torch.device(get(cfg, "device", default="cuda") if torch.cuda.is_available() else "cpu")

    output_dir = Path(get(cfg, "train", "output_dir", default="./runs/unistylediff_stage1"))
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

    dataset = ImageStyleDataset(
        content_dir=get(cfg, "data", "content_dir"),
        style_dir=get(cfg, "data", "style_dir"),
        image_size=get(cfg, "data", "image_size", default=512),
        crop_size=get(cfg, "data", "crop_size", default=256),
    )
    loader = DataLoader(dataset, batch_size=get(cfg, "train", "batch_size", default=8), shuffle=True, num_workers=get(cfg, "train", "num_workers", default=8))

    optim = torch.optim.AdamW(image_model.parameters(), lr=get(cfg, "train", "lr", default=1e-4))

    content_drop = get(cfg, "train", "content_drop", default=0.2)
    style_drop = get(cfg, "train", "style_drop", default=0.4)

    image_model.train()
    max_steps = get(cfg, "train", "max_steps", default=None)
    global_step = 0
    for epoch in range(get(cfg, "train", "epochs", default=30)):
        for step, (content, style) in enumerate(loader):
            content = content.to(device)
            style = style.to(device)

            latents = image_model.encode_latents(content)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            if random.random() < content_drop:
                content_cond = None
            else:
                content_cond = content

            if random.random() < style_drop:
                style_cond = None
            else:
                style_cond = style

            noise_pred = image_model(noisy_latents, timesteps, content=content_cond, style=style_cond)
            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            optim.zero_grad()
            loss.backward()
            optim.step()

            if (step + 1) % get(cfg, "train", "log_every", default=50) == 0:
                print(f"epoch {epoch} step {step+1}: loss={loss.item():.4f}")

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                ckpt_path = output_dir / f"stage1_step_{global_step}.pt"
                torch.save({"unet": unet.state_dict()}, ckpt_path)
                return

        ckpt_path = output_dir / f"stage1_epoch_{epoch+1}.pt"
        torch.save({"unet": unet.state_dict()}, ckpt_path)


if __name__ == "__main__":
    main()
