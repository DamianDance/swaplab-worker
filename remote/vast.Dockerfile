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
  && python facefusion.py force-download --download-scope lite --download-providers github huggingface --log-level info

RUN python -m pip install fastapi uvicorn python-multipart

COPY remote/gpu_worker.py /workspace/gpu_worker.py
WORKDIR /workspace
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "gpu_worker:app", "--host", "0.0.0.0", "--port", "7860"]
