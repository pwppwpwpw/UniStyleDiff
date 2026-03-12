UniStyleDiff Implementation

This directory contains an implementation aligned to:
UniStyleDiff: A unified diffusion-driven framework for image and video style transfer (ESWA 2026).

Key components:
- Stage I: image stylization with dual-branch adaptive feature injection.
- Stage II: video extension with a pluggable Inter-frame Consistency Module (ICM).
- Motion-Dynamics Preserved (MDP) sampling for temporal guidance.

Entry points:
- unistylediff/scripts/train_stage1.py
- unistylediff/scripts/train_stage2.py
- unistylediff/scripts/infer_image.py
- unistylediff/scripts/infer_video.py

Configs:
- configs/unistylediff/stage1.json
- configs/unistylediff/stage2.json
