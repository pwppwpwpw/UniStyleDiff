import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from unistylediff.pipelines.pipeline_image import UniStyleDiffImageModel
from unistylediff.utils.config import load_config, get


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(get(cfg, "device", default="cuda") if torch.cuda.is_available() else "cpu")

    try:
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    except ImportError as exc:
        raise ImportError("diffusers is required for inference") from exc

    model_id = get(cfg, "model", "pretrained", default="runwayml/stable-diffusion-v1-5")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

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

    tf = transforms.Compose(
        [
            transforms.Resize(get(cfg, "data", "image_size", default=512)),
            transforms.ToTensor(),
        ]
    )
    content = tf(Image.open(args.content).convert("RGB")).unsqueeze(0).to(device)
    style = tf(Image.open(args.style).convert("RGB")).unsqueeze(0).to(device)

    latents = torch.randn((1, 4, content.shape[-2] // 8, content.shape[-1] // 8), device=device)
    scheduler.set_timesteps(args.steps)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = image_model(latents, t, content=content, style=style)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = image_model.vae.decode(latents / image_model.vae.config.scaling_factor).sample
        images = (images.clamp(-1, 1) + 1) / 2.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(out_path)


if __name__ == "__main__":
    main()
