import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from unistylediff.pipelines.pipeline_image import UniStyleDiffImageModel
from unistylediff.pipelines.pipeline_video import UniStyleDiffVideoModel
from unistylediff.pipelines.mdp_sampler import MDPSampler
from unistylediff.utils.config import load_config, get


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
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

    video_model = UniStyleDiffVideoModel(
        image_model=image_model,
        icm_dim=get(cfg, "icm", "dim", default=320),
        icm_heads=get(cfg, "icm", "heads", default=8),
        icm_layers=get(cfg, "icm", "layers", default=2),
        icm_targets=get(cfg, "icm", "targets", default=[]),
    ).to(device)

    tf = transforms.Compose(
        [
            transforms.Resize(get(cfg, "data", "image_size", default=512)),
            transforms.ToTensor(),
        ]
    )

    frames = sorted([p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    video = torch.stack([tf(Image.open(p).convert("RGB")) for p in frames], dim=0).to(device)
    style = tf(Image.open(args.style).convert("RGB")).unsqueeze(0).to(device)

    f, _, h, w = video.shape
    latents = torch.randn((f, 4, h // 8, w // 8), device=device)
    ref_latents = image_model.encode_latents(video)

    sampler = MDPSampler(video_model, scheduler, guidance_scale=get(cfg, "mdp", "guidance_scale", default=1.0), time_scale=get(cfg, "mdp", "time_scale", default=1.0))
    scheduler.set_timesteps(args.steps)

    with torch.no_grad():
        for t in scheduler.timesteps:
            latents = sampler.step(
                latents,
                t,
                num_frames=f,
                content=video,
                style=style,
                reference_latents=ref_latents,
            )

        images = image_model.vae.decode(latents / image_model.vae.config.scaling_factor).sample
        images = (images.clamp(-1, 1) + 1) / 2.0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(f):
        Image.fromarray((images[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")).save(out_dir / f"{i:05d}.png")


if __name__ == "__main__":
    main()
