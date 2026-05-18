#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


FACEFUSION_DIR = Path(os.environ.get("FACEFUSION_DIR", "/opt/facefusion")).expanduser().resolve()
WORK_DIR = Path(os.environ.get("SWAPLAB_WORK_DIR", "/tmp/swaplab-gpu-worker")).expanduser().resolve()
TOKEN = os.environ.get("SWAPLAB_WORKER_TOKEN", "")

app = FastAPI(title="SwapLab GPU Worker")


@app.get("/health")
def health():
    return {
        "ok": True,
        "facefusionDir": str(FACEFUSION_DIR),
        "hasFaceFusion": (FACEFUSION_DIR / "facefusion.py").exists(),
        "gpu": gpu_name()
    }


@app.post("/process")
async def process(
    photo: UploadFile = File(...),
    video: UploadFile = File(...),
    job_id: str = Form(default="job"),
    processors: str = Form(default="face_swapper face_enhancer"),
    execution_providers: str = Form(default="cuda cpu"),
    swapper_model: str = Form(default="inswapper_128_fp16"),
    pixel_boost: str = Form(default="512x512"),
    enhancer_model: str = Form(default="gfpgan_1.4"),
    enhancer_blend: str = Form(default="80"),
    selector_mode: str = Form(default="many"),
    video_encoder: str = Form(default="libx264"),
    video_quality: str = Form(default="90"),
    video_preset: str = Form(default="veryfast"),
    authorization: Optional[str] = Header(default=None)
):
    verify_token(authorization)
    ensure_facefusion()

    safe_job_id = "".join(char for char in job_id if char.isalnum() or char in "-_")[:80] or "job"
    job_dir = WORK_DIR / f"{safe_job_id}-{uuid.uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=True)
    photo_path = job_dir / safe_name(photo.filename or "source.jpg")
    video_path = job_dir / safe_name(video.filename or "target.mp4")
    output_path = job_dir / "result.mp4"
    temp_path = job_dir / "temp"

    try:
        await save_upload(photo, photo_path)
        await save_upload(video, video_path)

        command = [
            sys.executable,
            str(FACEFUSION_DIR / "facefusion.py"),
            "headless-run",
            "--source-paths", str(photo_path),
            "--target-path", str(video_path),
            "--output-path", str(output_path),
            "--temp-path", str(temp_path),
            "--processors", *split_words(processors),
            "--face-swapper-model", swapper_model,
            "--face-swapper-pixel-boost", pixel_boost,
            "--face-enhancer-model", enhancer_model,
            "--face-enhancer-blend", enhancer_blend,
            "--face-selector-mode", selector_mode,
            "--execution-providers", *split_words(execution_providers),
            "--output-video-encoder", video_encoder,
            "--output-video-quality", video_quality,
            "--output-video-preset", video_preset,
            "--log-level", "info"
        ]
        subprocess.run(command, cwd=FACEFUSION_DIR, check=True, timeout=int(os.environ.get("SWAPLAB_WORKER_TIMEOUT_SECONDS", "3600")))
        if not output_path.exists():
            raise RuntimeError("FaceFusion finished without producing result.mp4.")
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="result.mp4",
            background=BackgroundTask(shutil.rmtree, job_dir, ignore_errors=True)
        )
    except subprocess.CalledProcessError as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"FaceFusion failed with exit code {error.returncode}.") from error
    except Exception as error:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(error)) from error


def verify_token(authorization: Optional[str]):
    if not TOKEN:
        return
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid worker token.")


def ensure_facefusion():
    if not (FACEFUSION_DIR / "facefusion.py").exists():
        raise HTTPException(status_code=500, detail=f"FaceFusion not found at {FACEFUSION_DIR}.")


async def save_upload(upload: UploadFile, destination: Path):
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def safe_name(name: str):
    clean = "".join(char if char.isalnum() or char in ".-_" else "_" for char in name)
    return clean[:120] or "upload.bin"


def split_words(value: str):
    return [part for part in value.split() if part]


def gpu_name():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5
        )
        return output.strip().splitlines()[0] if output.strip() else None
    except Exception:
        return None
