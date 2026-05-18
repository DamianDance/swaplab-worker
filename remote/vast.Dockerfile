FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG FACEFUSION_REF=3.6.1
ENV DEBIAN_FRONTEND=noninteractive \
    FACEFUSION_DIR=/opt/facefusion \
    SWAPLAB_WORK_DIR=/workspace/jobs \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates ffmpeg git \
  && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
  && git clone --branch "${FACEFUSION_REF}" --depth 1 https://github.com/facefusion/facefusion.git /opt/facefusion \
  && cd /opt/facefusion \
  && python -m pip install -r requirements.txt \
  && python install.py --onnxruntime cuda --skip-conda \
  && (python facefusion.py force-download \
    --processors face_swapper face_enhancer \
    --face-swapper-model inswapper_128_fp16 \
    --face-swapper-pixel-boost 512x512 \
    --face-enhancer-model gfpgan_1.4 || true)

RUN python -m pip install fastapi uvicorn python-multipart

COPY remote/gpu_worker.py /workspace/gpu_worker.py
WORKDIR /workspace
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "gpu_worker:app", "--host", "0.0.0.0", "--port", "7860"]
