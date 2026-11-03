# syntax=docker/dockerfile:1
# Nemotron 3.5 Transcribe 的 vLLM 服務映像，build context 是 repo 根目錄。
# 相依版本全由 repo 的 uv.lock 決定（asr + vllm extras），與模型卡的安裝步驟相同；
# vllm extra 不能跟 cu12/cu13/all 混裝，vLLM 自帶它要的 torch 與 CUDA kernel。
#
# 兩個 stage 都用 CUDA devel：FlashInfer 第一次執行量化 attention / Mamba kernel 時
# 要用 nvcc 現場編譯（GB10 是 SM121，沒有預編好的 kernel），runner 少了 nvcc 會在第一個請求掛掉。
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04

FROM ${CUDA_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.27 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=copy

WORKDIR /opt/nemo-speech

# 先只放鎖檔裝相依（約 10 GB），之後改 nemo/ 原始碼不必重下這一層
COPY pyproject.toml uv.lock setup.py README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --python 3.13 --extra asr --extra vllm --locked --no-install-project

COPY nemo ./nemo
# nemo 本體以 editable 裝，vLLM 才找得到 nemo_speechlm plugin 的 entry point。
# PyAV 解 m4a 等壓縮格式不必系統 FFmpeg；vLLM 需要 OpenCV，但 NeMo 的相依設定把它排除了。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps -e . \
    && uv pip install --python /opt/venv/bin/python --no-deps \
        av==19.0.2 opencv-python-headless==5.0.0.93

FROM ${CUDA_IMAGE} AS runner

# Triton / FlashInfer JIT 編 kernel launcher 時要 C 編譯器
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/nemo-speech /opt/nemo-speech

ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1

COPY --chmod=755 deploy/nemotron-transcribe/entrypoint.sh /usr/local/bin/entrypoint.sh

WORKDIR /opt/nemo-speech
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
