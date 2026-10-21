# Dockerfile na RAIZ do repositório (ao lado do requirements.txt).
# Código da demo em src/demo/. Requer GPU NVIDIA no host.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-tk ffmpeg git libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# vllm primeiro (fixa a versão do torch); depois o requirements da raiz e o backports.tarfile
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install vllm \
    && pip install -r requirements.txt \
    && pip install --upgrade backports.tarfile

# só o código da demo (e a pasta tools/, se estiver dentro dela)
COPY src/demo ./src/demo
RUN mkdir -p data/evidence

# evidências ficam no volume montado em /app/data/evidence
ENV MAIDAI_EVIDENCE_DIR=/app/data/evidence

ENTRYPOINT ["python", "src/demo/main.py"]
CMD ["--no-gui"]
