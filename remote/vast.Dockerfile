FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG FACEFUSION_REF=3.6.1
ENV DEBIAN_FRONTEND=noninteractive \
    FACEFUSION_DIR=/opt/facefusion \
    SWAPLAB_WORK_DIR=/workspace/jobs \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_SITE_PACKAGES=/opt/conda/lib/python3.11/site-packages/nvidia \
    LD_LIBRARY_PATH=/opt/conda/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cufft/lib:/opt/conda/lib/python3.11/site-packages/nvidia/curand/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda/lib64

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg git \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
  && python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-nvrtc-cu12 \
  && git clone --branch "${FACEFUSION_REF}" --depth 1 https://github.com/facefusion/facefusion.git /opt/facefusion \
  && cd /opt/facefusion \
  && python -m pip install -r requirements.txt \
  && python install.py --onnxruntime cuda --skip-conda \
  && python -c "import ctypes, onnxruntime as ort; [ctypes.CDLL(lib) for lib in ('libcublasLt.so.12', 'libcudnn.so.9')]; providers = ort.get_available_providers(); print('ONNX Runtime providers:', providers); assert 'CUDAExecutionProvider' in providers" \
  && mkdir -p /tmp/swaplab-warmup \
  && ffmpeg -y -f lavfi -i color=c=black:s=320x320 -frames:v 1 /tmp/swaplab-warmup/source.jpg >/dev/null 2>&1 \
  && ffmpeg -y -f lavfi -i color=c=black:s=320x320:r=1:d=0.2 -pix_fmt yuv420p /tmp/swaplab-warmup/target.mp4 >/dev/null 2>&1 \
  && for spec in "inswapper_128_fp16 512x512" "hyperswap_1a_256 512x512" "simswap_unofficial_512 512x512"; do \
    set -- $spec; \
    python facefusion.py headless-run \
      --source-paths /tmp/swaplab-warmup/source.jpg \
      --target-path /tmp/swaplab-warmup/target.mp4 \
      --output-path /tmp/swaplab-warmup/result.mp4 \
      --temp-path /tmp/swaplab-warmup/temp \
      --processors face_swapper face_enhancer \
      --face-swapper-model "$1" \
      --face-swapper-pixel-boost "$2" \
      --face-enhancer-model gfpgan_1.4 \
      --execution-providers cpu \
      --output-video-encoder libx264 \
      --output-video-quality 70 \
      --output-video-preset ultrafast \
      --log-level error || true; \
  done \
  && test -f /opt/facefusion/.assets/models/inswapper_128_fp16.onnx \
  && test -f /opt/facefusion/.assets/models/hyperswap_1a_256.onnx \
  && test -f /opt/facefusion/.assets/models/simswap_unofficial_512.onnx \
  && test -f /opt/facefusion/.assets/models/gfpgan_1.4.onnx

RUN python -m pip install fastapi uvicorn python-multipart

COPY remote/gpu_worker.py /workspace/gpu_worker.py
WORKDIR /workspace
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "gpu_worker:app", "--host", "0.0.0.0", "--port", "7860"]
