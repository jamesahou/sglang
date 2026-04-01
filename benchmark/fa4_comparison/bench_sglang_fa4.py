import json
import os
import statistics
import time
from random import randint, seed

from sglang.srt.entrypoints.engine import Engine

MODEL = "Qwen/Qwen3-14B"
NUM_SEQS = 512
NUM_TRIALS = 5
NUM_WARMUP = 2
MAX_INPUT_LEN = 1024
MAX_OUTPUT_LEN = 1024

# ATTN_BACKEND uses shared naming convention across all three benchmarks:
#   fa4      -> fa4
#   fa3 / fa -> fa3
#   fi       -> flashinfer
#   trtllm   -> trtllm_mha
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "fa4")
RESULT_FILE = f"sglang_{ATTN_BACKEND.replace(',', '_')}_result.json"

_BACKEND_MAP = {
    "fa4":    "fa4",
    "fa3":    "fa3",
    "fa":     "fa3",
    "fi":     "flashinfer",
    "trtllm": "trtllm_mha",
}


def sglang_backend_name(backend: str) -> str:
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown ATTN_BACKEND '{backend}'. "
            f"Valid options: {list(_BACKEND_MAP.keys())}"
        )
    return _BACKEND_MAP[backend]


def make_inputs(rng_seed: int):
    # Same seed and length distribution as the other two benchmark scripts
    seed(rng_seed)
    input_ids = [
        [randint(0, 10000) for _ in range(randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    max_tokens_list = [randint(100, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    sampling_params = [
        {"temperature": 0.6, "max_new_tokens": n, "ignore_eos": True}
        for n in max_tokens_list
    ]
    return input_ids, sampling_params, max_tokens_list


def run_batch(engine, input_ids, sampling_params):
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    return outputs


def main():
    sgl_backend = sglang_backend_name(ATTN_BACKEND)
    print(f"Loading {MODEL} with backend: {ATTN_BACKEND} (sglang: {sgl_backend})")

    engine = Engine(
        model_path=MODEL,
        attention_backend=sgl_backend,
        context_length=4096,
        mem_fraction_static=0.9,
        cuda_graph_max_bs=NUM_SEQS,
    )

    input_ids, sampling_params, max_tokens_list = make_inputs(rng_seed=0)
    total_tokens = sum(max_tokens_list)

    print(f"Warming up ({NUM_WARMUP} batches)...")
    for i in range(NUM_WARMUP):
        run_batch(engine, input_ids, sampling_params)
        print(f"  warmup {i + 1}/{NUM_WARMUP} done")

    print(f"Running {NUM_TRIALS} timed trials ({NUM_SEQS} seqs, {total_tokens} total output tokens each)...")
    times = []
    for i in range(NUM_TRIALS):
        t = time.perf_counter()
        run_batch(engine, input_ids, sampling_params)
        elapsed = time.perf_counter() - t
        times.append(elapsed)
        print(f"  trial {i + 1}/{NUM_TRIALS}: {elapsed:.2f}s  ({total_tokens / elapsed:.1f} tok/s)")

    # Discard min and max, compute stats on remaining
    times_trimmed = sorted(times)[1:-1]
    throughputs = [total_tokens / t for t in times_trimmed]
    mean_tp = statistics.mean(throughputs)
    std_tp = statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0

    print(f"\n=== SGLang {ATTN_BACKEND} Results ===")
    print(f"  Throughput (trimmed mean): {mean_tp:.1f} tok/s")
    print(f"  Throughput std:            {std_tp:.1f} tok/s")
    print(f"  All trial throughputs:     {[f'{x:.1f}' for x in [total_tokens / t for t in times]]}")

    result = {
        "system": "sglang",
        "backend": ATTN_BACKEND,
        "model": MODEL,
        "num_seqs": NUM_SEQS,
        "total_tokens": total_tokens,
        "num_trials": NUM_TRIALS,
        "num_warmup": NUM_WARMUP,
        "throughput_mean": round(mean_tp, 2),
        "throughput_std": round(std_tp, 2),
        "trial_throughputs": [round(total_tokens / t, 2) for t in times],
        "all_times_s": [round(t, 4) for t in times],
    }
    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {RESULT_FILE}")

    engine.shutdown()


if __name__ == "__main__":
    main()
