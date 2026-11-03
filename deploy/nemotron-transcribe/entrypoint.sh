#!/usr/bin/env bash
# 依這台的 GPU 自動決定精度與記憶體比例，模型不在就先下載，最後 exec vllm serve。
# CPU 架構（x86_64 / aarch64）在 build 時就由多架構 base image 與 uv.lock 決定，這裡只管 GPU。
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/models}
REPO_BASE=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B

read -r cc mem_mib < <(
    nvidia-smi --query-gpu=compute_cap,memory.total --format=csv,noheader,nounits \
        | head -1 | tr -d ' ' | tr ',' ' '
)
cc_major=${cc%%.*}
cc_num=$(( cc_major * 10 + ${cc#*.} ))

# GB10 這類統一記憶體的機器 nvidia-smi 回報不出顯存，vLLM 看到的是整台的 RAM，
# 而且跟宿主上其他服務共用，比例要壓低
if [[ ! "$mem_mib" =~ ^[0-9]+$ ]]; then
    unified=1
    mem_mib=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 ))
else
    unified=0
fi

# 模型卡：FP8 要 SM89 以上、NVFP4 要 Blackwell（SM100/SM12x）。
# auto 時 SM89 以上選 FP8（準確度與 BF16 相同、權重少一半），更舊的卡只能跑 BF16。
variant=${MODEL_VARIANT:-auto}
if [[ "$variant" == auto ]]; then
    if (( cc_num >= 89 )); then variant=fp8; else variant=bf16; fi
fi
case "$variant" in
    bf16) repo=$REPO_BASE; weights_gib=63 ;;
    fp8) repo=$REPO_BASE-FP8; weights_gib=34 ;;
    nvfp4) repo=$REPO_BASE-NVFP4; weights_gib=21 ;;
    *) echo "MODEL_VARIANT 必須是 auto/bf16/fp8/nvfp4，收到 '$variant'" >&2; exit 1 ;;
esac
if [[ "$variant" == fp8 ]] && (( cc_num < 89 )); then
    echo "FP8 需要 SM89 以上的 GPU，這張是 SM$cc_num" >&2; exit 1
fi
if [[ "$variant" == nvfp4 ]] && (( cc_num < 100 )); then
    echo "NVFP4 需要 Blackwell（SM100/SM12x），這張是 SM$cc_num" >&2; exit 1
fi

# vLLM 啟動時就把整個 pool 佔下來，不隨負載伸縮。統一記憶體跟宿主上所有服務共用，
# 只給權重加 CACHE_GIB（Mamba 狀態、KV cache、CUDA graph、activation），不照比例吃。
cache_gib=${CACHE_GIB:-9}
if [[ -z "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    if (( unified )); then
        GPU_MEMORY_UTILIZATION=$(awk -v w="$weights_gib" -v c="$cache_gib" -v m="$mem_mib" \
            'BEGIN{printf "%.2f", (w+c)*1024/m}')
    else
        GPU_MEMORY_UTILIZATION=0.90
    fi
fi
budget_gib=$(awk -v m="$mem_mib" -v u="$GPU_MEMORY_UTILIZATION" 'BEGIN{printf "%d", m*u/1024}')
# 權重之外至少要留幾 GB 給 Mamba 狀態、KV cache 與 CUDA graph，否則 vLLM 載完權重才報錯
if (( budget_gib < weights_gib + 4 )); then
    echo "記憶體不夠：$variant 權重約 ${weights_gib} GiB，" \
         "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION 只給 ${budget_gib} GiB" \
         "（總共 $(( mem_mib / 1024 )) GiB）。改用較小的 MODEL_VARIANT 或調高比例。" >&2
    exit 1
fi

# 統一記憶體上 GPU 配置不算進任何 process 的 RSS，OOM killer 找不到真正的大戶，
# 只會一路殺小 process 直到整台失聯（2026-10-30 在 GB10 上發生過，只能斷電重開）。
# 所以開機前先確認「現在」可用的記憶體夠：pool 加上 vLLM 的 CPU 端與 FlashInfer JIT 編譯。
if (( unified )); then
    headroom_gib=${HEADROOM_GIB:-16}
    avail_gib=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
    need_gib=$(( budget_gib + headroom_gib ))
    if (( avail_gib < need_gib )); then
        echo "可用記憶體只有 ${avail_gib} GiB，需要 ${need_gib} GiB" \
             "（pool ${budget_gib} + 餘裕 ${headroom_gib}）。不啟動，以免拖垮整台。" \
             "先關掉其他服務，或改 MODEL_VARIANT=nvfp4。" >&2
        # 停一下再退出，避免 restart 策略在記憶體不夠時密集重試
        sleep "${INSUFFICIENT_MEMORY_BACKOFF_S:-300}"
        exit 1
    fi
fi

model_dir=$MODEL_ROOT/${repo#*/}
if [[ ! -f "$model_dir/config.json" ]]; then
    if [[ "${AUTO_DOWNLOAD:-1}" != 1 ]]; then
        echo "找不到 $model_dir，且 AUTO_DOWNLOAD=0" >&2; exit 1
    fi
    echo "下載 $repo 到 $model_dir"
    hf download "$repo" --local-dir "$model_dir"
fi

# SM12x（GB10、RTX PRO 6000、RTX 50 系列）上 FlashInfer 的 FP8 GEMM 跑到一半可能崩潰，
# 模型卡的 FP8 設定也把它關掉改用 CUTLASS
if (( cc_major == 12 )) && [[ "$variant" != bf16 ]]; then
    export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel}
fi
# NVFP4 的 ModelOpt 匯出沒宣告 KV cache 格式，auto 會留在 BF16
kv_dtype=auto
[[ "$variant" == nvfp4 ]] && kv_dtype=fp8

# NUM_SPECULATIVE_TOKENS=0 關掉 MTP 推測解碼（交叉測試用）；ReplaySSM 只在驗證 draft 時需要
spec_args=()
if (( ${NUM_SPECULATIVE_TOKENS:-8} > 0 )); then
    spec_args=(--use-replayssm
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS:-8}}")
fi

echo "GPU SM$cc_num，$(( mem_mib / 1024 )) GiB$( (( unified )) && echo '（統一記憶體）')；" \
     "精度 $variant，gpu-memory-utilization $GPU_MEMORY_UTILIZATION（約 ${budget_gib} GiB）"

exec vllm serve "$model_dir" \
    --host 0.0.0.0 --port 8000 \
    --served-model-name "${SERVED_MODEL_NAME:-nemotron-transcribe}" \
    --trust-remote-code --dtype bfloat16 --kv-cache-dtype "$kv_dtype" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --max-num-batched-tokens 4096 \
    --max-num-seqs "${MAX_NUM_SEQS:-16}" \
    --enable-chunked-prefill --no-enable-prefix-caching \
    --mamba-backend flashinfer --mamba-cache-mode none \
    --mm-processor-cache-gb 0 --no-async-scheduling \
    --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
    "${spec_args[@]}" \
    --reasoning-parser nemotron_v3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --trust-request-mm-kwargs \
    --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
    "$@"
