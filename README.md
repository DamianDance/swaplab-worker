# SwapLab Vast Worker

Minimal public build context for the SwapLab Vast.ai GPU worker image.

This repository intentionally contains only:

- `remote/gpu_worker.py`
- `remote/vast.Dockerfile`
- `.dockerignore`
- GitHub Actions workflow for GHCR publishing

It does not contain local uploads, videos, photos, `.env`, API keys, jobs, or the app database.
