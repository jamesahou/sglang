#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

nsys profile \
    --trace=cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output=profile_sglang \
    --force-overwrite=true \
    uv run benchmark.py --config configs/decode_bs128.yaml --repeats 5 --warmup-iters 3 --batch-specs 16q1s60000
