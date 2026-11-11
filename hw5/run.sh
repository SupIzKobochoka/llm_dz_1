#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    NVCC="$(find "$CONDA_PREFIX" -type f -name nvcc -print -quit)"
    if [[ -n "$NVCC" ]]; then
        export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
fi

if [[ ! -f data/train.jsonl || ! -f data/val.jsonl ]]; then
    "$PYTHON" prepare.py
fi

if [[ ! -f results/base/metrics.json ]]; then
    "$PYTHON" evaluate.py Qwen/Qwen3.5-2B --output results/base
fi

for run in rl1 rl2 rl3; do
    "$PYTHON" train.py "$run" --steps 60
    "$PYTHON" evaluate.py "./checkpoints/$run" --output "results/$run"
done


echo "Готово"