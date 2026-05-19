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

ESSENTIAL_MODELS = [
    "yoloface_8n.onnx",
    "fan_68_5.onnx",
    "2dfan4.onnx",
    "xseg_1.onnx",
    "bisenet_resnet_34.onnx",
    "arcface_w600k_r50.onnx",
    "inswapper_128_fp16.onnx",
    "hyperswap_1a_256.onnx",
    "simswap_unofficial_512.onnx",
    "crossface_simswap.onnx",
    "gfpgan_1.4.onnx",
]

PREFETCH_MODELS = [
    ("inswapper_128_fp16", "512x512"),
    ("hyperswap_1a_256", "512x512"),
    ("simswap_unofficial_512", "512x512"),
]


@app.get("/health")
def health():
    return {
        "ok": True,
        "facefusionDir": str(FACEFUSION_DIR),
        "hasFaceFusion": (FACEFUSION_DIR / "facefusion.py").exists(),
        "gpu": gpu_name(),
        "modelCache": model_cache_status()
    }


@app.post("/warmup")
def warmup(authorization: Optional[str] = Header(default=None)):
    verify_token(authorization)
    ensure_facefusion()
    prefetch_models()
    return {
        "ok": True,
        "gpu": gpu_name(),
        "modelCache": model_cache_status()
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
    enhancer_weight: str = Form(default="0.55"),
    swapper_weight: str = Form(default="0.55"),
    selector_mode: str = Form(default="many"),
    video_encoder: str = Form(default="libx264"),
    video_quality: str = Form(default="90"),
    video_preset: str = Form(default="veryfast"),
    output_video_fps: str = Form(default="0"),
    execution_thread_count: str = Form(default="8"),
    video_memory_strategy: str = Form(default="tolerant"),
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
            "--face-swapper-weight", swapper_weight,
            "--face-enhancer-model", enhancer_model,
            "--face-enhancer-blend", enhancer_blend,
            "--face-enhancer-weight", enhancer_weight,
            "--face-selector-mode", selector_mode,
            "--execution-providers", *split_words(execution_providers),
            "--execution-thread-count", execution_thread_count,
            "--video-memory-strategy", video_memory_strategy,
            "--output-video-encoder", video_encoder,
            "--output-video-quality", video_quality,
            "--output-video-preset", video_preset,
            "--log-level", "info"
        ]
        if as_positive_number(output_video_fps) > 0:
            command.extend(["--output-video-fps", output_video_fps])
        run_facefusion(command, video_encoder, output_path)
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


def prefetch_models():
    warmup_dir = WORK_DIR / "_warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    source_path = warmup_dir / "source.jpg"
    target_path = warmup_dir / "target.mp4"
    output_path = warmup_dir / "result.mp4"
    temp_path = warmup_dir / "temp"
    timeout = int(os.environ.get("SWAPLAB_WARMUP_TIMEOUT_SECONDS", "900"))

    if not source_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x320",
            "-frames:v", "1", str(source_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if not target_path.exists():
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x320:r=1:d=0.2",
            "-pix_fmt", "yuv420p", str(target_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)

    for swapper_model, pixel_boost in PREFETCH_MODELS:
        if output_path.exists():
            output_path.unlink()
        shutil.rmtree(temp_path, ignore_errors=True)
        command = [
            sys.executable,
            str(FACEFUSION_DIR / "facefusion.py"),
            "headless-run",
            "--source-paths", str(source_path),
            "--target-path", str(target_path),
            "--output-path", str(output_path),
            "--temp-path", str(temp_path),
            "--processors", "face_swapper", "face_enhancer",
            "--face-swapper-model", swapper_model,
            "--face-swapper-pixel-boost", pixel_boost,
            "--face-enhancer-model", "gfpgan_1.4",
            "--execution-providers", "cpu",
            "--output-video-encoder", "libx264",
            "--output-video-quality", "70",
            "--output-video-preset", "ultrafast",
            "--log-level", "error",
        ]
        try:
            subprocess.run(command, cwd=FACEFUSION_DIR, check=True, timeout=timeout)
        except subprocess.CalledProcessError:
            pass


def run_facefusion(command, video_encoder: str, output_path: Path):
    timeout = int(os.environ.get("SWAPLAB_WORKER_TIMEOUT_SECONDS", "3600"))
    try:
        subprocess.run(command, cwd=FACEFUSION_DIR, check=True, timeout=timeout)
        return
    except subprocess.CalledProcessError:
        if video_encoder not in {"h264_nvenc", "hevc_nvenc"}:
            raise
        if output_path.exists():
            output_path.unlink()
        fallback = replace_arg(command, "--output-video-encoder", "libx264")
        fallback = replace_arg(fallback, "--output-video-preset", "veryfast")
        subprocess.run(fallback, cwd=FACEFUSION_DIR, check=True, timeout=timeout)


def replace_arg(command, name: str, value: str):
    next_command = list(command)
    try:
        index = next_command.index(name)
    except ValueError:
        return next_command + [name, value]
    if index + 1 < len(next_command):
        next_command[index + 1] = value
    return next_command


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


def as_positive_number(value: str):
    try:
        number = float(value)
    except Exception:
        return 0
    return number if number > 0 else 0


def model_cache_status():
    models_dir = FACEFUSION_DIR / ".assets" / "models"
    missing = [name for name in ESSENTIAL_MODELS if not (models_dir / name).exists()]
    return {
        "ready": len(missing) == 0,
        "missing": missing[:12]
    }


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
