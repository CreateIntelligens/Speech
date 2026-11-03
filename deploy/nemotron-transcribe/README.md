# Nemotron 3.5 Transcribe 部署（vLLM）

`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` 的 OpenAI 相容服務。語音 encoder 接 Nemotron 3.5 30B-A4B MoE 語言模型，輸入音訊加文字 prompt，輸出文字。x86_64 與 aarch64 共用同一份設定；目前只在 aarch64（GB10）實際跑過。

```bash
cd deploy/nemotron-transcribe
free -g                                 # 先看可用記憶體
docker compose build transcribe-api     # 單獨 build 完再 up，不要並行 build
docker compose up -d                    # 對外 http://<host>:8804（nginx）
python3 transcribe_test.py --prompt "請將這段音訊逐字轉寫成繁體中文，只輸出逐字稿。" a.wav
python3 transcribe_test.py --speakers --timestamps meeting.wav
```

`.env` 可設 `MODEL_VARIANT`（auto / bf16 / fp8 / nvfp4）、`PUBLIC_PORT`、`MAX_NUM_SEQS` 等，見 `.env.example`。模型不在 `models/` 時，啟動會自動下載。

## 統一記憶體機器

`--gpu-memory-utilization` 是整台記憶體的比例，而且啟動就規劃整塊。vLLM 預設 0.92、模型卡建議 0.85–0.94，都是給獨立顯卡的，在統一記憶體上等於吃掉整台機器。`entrypoint.sh` 改成依需要量反推：權重加 `CACHE_GIB`（預設 9）。

- `MAX_JOBS=2`：FlashInfer 第一次啟動要用 nvcc 現編 kernel，不限制時並行數高，記憶體可能撐爆。編好的 kernel 存在 `cache/`。
- `oom_score_adj: 1000`：OOM 時先殺這個容器。它只保護到第一個受害者，kernel 接著可能殺別的服務。
- 啟動前檢查 MemAvailable 要大於等於 pool 加 `HEADROOM_GIB`（16），不夠就拒絕啟動。
- 不要設 `NUM_SPECULATIVE_TOKENS=0`：實測啟動時記憶體會暴衝，原因未查。

## 使用上的注意事項

- 華語要用中文 prompt。英文 prompt 可能把華語翻成英文；繁簡不穩，要繁體請在後端再轉一次。
- 請求要帶 `max_tokens`：模型聽不懂時可能重複輸出到 token 用完。
- 回應可能是 HTTP 200、`finish=stop` 但內容空白，呼叫端要檢查。
- 字級時間戳要求 `stream=false`、單一音檔、`skip_special_tokens=false`。搭配串流時會被靜默忽略。
- 三人以上的講者標籤不穩定，同一段音檔每次結果可能差很多。
- 不是串流 ASR：要拿到完整音訊才開始處理。

## 測試腳本

`qa/` 是可重跑的測試框架，資料與結果都不進版控。

```bash
uv run --no-project --with numpy --with soundfile python qa/prepare.py
python3 qa/run_suite.py --variant <名稱>            # 可加 --only、--groups、--rounds
uv run --no-project --with jiwer --with opencc-python-reimplemented --with soundfile python qa/report.py <名稱>/<run id>
uv run --no-project --with jiwer --with opencc-python-reimplemented --with soundfile python -m unittest qa/test_checks.py
```

每輪結果存到新的目錄並附版本資訊，不會覆寫；`report.py` 預設拒絕不完整或版本不同的結果。
