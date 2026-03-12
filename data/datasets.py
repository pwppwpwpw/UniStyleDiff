from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def list_images(root: str) -> List[Path]:
    root_path = Path(root)
    return [p for p in root_path.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".webp"}]


class ImageStyleDataset(Dataset):
    def __init__(self, content_dir: str, style_dir: str, image_size: int = 512, crop_size: int = 256):
        self.content_paths = list_images(content_dir)
        self.style_paths = list_images(style_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomCrop(crop_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return max(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        content_path = self.content_paths[idx % len(self.content_paths)]
        style_path = self.style_paths[idx % len(self.style_paths)]
        content = self.transform(Image.open(content_path).convert("RGB"))
        style = self.transform(Image.open(style_path).convert("RGB"))
        return content, style


class VideoStyleDataset(Dataset):
    """
    Expects each video as a folder of frames.
    """

    def __init__(self, video_root: str, style_dir: str, num_frames: int = 16, frame_stride: int = 4, image_size: int = 512):
        self.video_dirs = [p for p in Path(video_root).iterdir() if p.is_dir()]
        self.style_paths = list_images(style_dir)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_dir = self.video_dirs[idx]
        frames = sorted(list_images(str(video_dir)))
        start = 0
        end = start + self.num_frames * self.frame_stride
        frames = frames[start:end:self.frame_stride]
        if len(frames) < self.num_frames:
            frames = (frames + frames[:1] * self.num_frames)[: self.num_frames]
        video = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in frames], dim=0)

        style_path = self.style_paths[idx % len(self.style_paths)]
        style = self.transform(Image.open(style_path).convert("RGB"))
        return video, style
