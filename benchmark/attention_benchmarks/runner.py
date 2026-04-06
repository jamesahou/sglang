"""
SGLang attention benchmark runner — backend approach.

Uses FlashAttentionBackend and TRTLLMHAAttnBackend directly via __new__ +
direct attribute assignment, bypassing the full model_runner init chain while
exercising the exact same init_forward_metadata + forward_decode/forward_extend
code paths as production.

Why __new__ instead of full mocking:
  FlashInferAttnBackend.__init__ (parent of TRTLLMHAAttnBackend) touches ~20
  model_runner attributes and does non-trivial init logic irrelevant to MHA
  benchmarking.  __new__ lets us set only the ~12 attributes each backend's
  methods actually read, without inheriting parent-class side effects.

KV cache layout:
  sglang stores KV as flat token slots: k_buffer[layer] shape [num_slots, nkv_h, head_dim]
  where num_slots = num_blocks * page_size.  The backend reshapes to
  [num_blocks, page_size, nkv_h, head_dim] internally.

req_to_token format:
  req_to_token[i, pos] = flat_slot_index (block * page_size + within_block).
  init_forward_metadata converts to block indices via:
    page_table = req_to_token[:, ::page_size] // page_size
"""

import math
from types import SimpleNamespace

import numpy as np
import torch
from batch_spec import parse_batch_spec
from common import BenchmarkConfig, BenchmarkResult, get_attention_scale

_WORKSPACE_BYTES = 512 * 1024 * 1024  # matches DEFAULT_WORKSPACE_SIZE_MB in trtllm_mha_backend.py


# ---------------------------------------------------------------------------
# Mock KV pool
# ---------------------------------------------------------------------------

class _MockKVPool:
    """
    Minimal token-to-KV-pool that returns pre-allocated flat slot buffers.

    Shape per layer: [num_slots, num_kv_heads, head_dim]
    where num_slots = num_blocks * block_size.

    The backends internally reshape via:
      FA:    key_cache.view(-1, page_size, nkv_h, head_dim)   → NHD blocks
      TRTLLM: same view then .permute(0, 2, 1, 3)             → HND blocks
    """

    def __init__(self, k_buffers: list, v_buffers: list):
        self._k = k_buffers
        self._v = v_buffers

    def get_kv_buffer(self, layer_id: int):
        return self._k[layer_id], self._v[layer_id]

    def set_kv_buffer(self, *args, **kwargs):
        pass  # KV write excluded from benchmark (same methodology as vLLM)


# ---------------------------------------------------------------------------
# Tensor construction helpers
# ---------------------------------------------------------------------------

def _build_req_to_token(
    kv_lens: list[int],
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build req_to_token in sglang's native flat-slot-index format.

    req_to_token[i, pos] = block_index * block_size + within_block_offset

    This is what init_forward_metadata reads from req_to_token_pool.req_to_token.
    It then converts to block-level page_table via:
        strided_indices = arange(0, max_kv, block_size)
        page_table = req_to_token[:, strided_indices] // block_size
    """
    batch_size = len(kv_lens)
    max_kv = max(kv_lens)
    max_blocks_per_req = math.ceil(max_kv / block_size)

    # Request i occupies blocks [i*B, (i+1)*B) contiguously
    base_blocks = torch.arange(batch_size, device=device).unsqueeze(1) * max_blocks_per_req
    positions   = torch.arange(max_kv, device=device).unsqueeze(0)

    req_to_token = (
        (base_blocks + positions // block_size) * block_size + positions % block_size
    ).to(torch.int32)

    return req_to_token


def _create_flat_kv_buffers(
    config: BenchmarkConfig,
    num_slots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list, list]:
    """Allocate KV buffers in sglang's native [num_slots, num_kv_heads, head_dim] layout."""
    k_buffers = [
        torch.zeros(num_slots, config.num_kv_heads, config.head_dim, dtype=dtype, device=device)
        for _ in range(config.num_layers)
    ]
    v_buffers = [
        torch.zeros(num_slots, config.num_kv_heads, config.head_dim, dtype=dtype, device=device)
        for _ in range(config.num_layers)
    ]
    return k_buffers, v_buffers


def _create_input_tensors(
    config: BenchmarkConfig,
    total_q: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list:
    return [
        torch.randn(total_q, config.num_q_heads, config.head_dim, dtype=dtype, device=device)
        for _ in range(config.num_layers)
    ]


# ---------------------------------------------------------------------------
# Backend construction via __new__
# ---------------------------------------------------------------------------

def _make_fa_backend(config: BenchmarkConfig, device: torch.device):
    """
    Construct FlashAttentionBackend by setting only the attributes read by
    init_forward_metadata + forward_decode/forward_extend on a standard
    non-speculative, non-SWA, non-MLA batch.

    num_splits mirrors FlashAttentionBackend.__init__ ~L390:
      FA4 + CUDA graphs → 1  (FA4 does not support num_splits=0 with CUDA Graph)
      otherwise         → 0  (let FA heuristic decide)
    """
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

    fa_ver = 3 if config.backend.lower() == "fa3" else 4
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)

    backend.fa_impl_ver                    = fa_ver
    backend.num_splits                     = 1 if (fa_ver == 4 and config.use_cuda_graphs) else 0
    backend.page_size                      = config.block_size
    backend.kv_cache_dtype_str             = "auto"
    backend.kv_cache_dtype                 = config.dtype
    backend.use_mla                        = False
    backend.has_local_attention            = False
    backend.has_swa                        = False
    backend.sliding_window_size            = None
    backend.use_sliding_window_kv_pool     = False
    backend.is_encoder_decoder             = False
    backend.device                         = device
    backend.topk                           = 0
    backend.speculative_step_id            = 0
    backend.speculative_num_steps          = 0
    backend.speculative_num_draft_tokens   = 0
    backend.skip_prefill                   = False
    backend.forward_metadata               = None
    backend.forward_metadata_spec_decode_expand = None
    backend.decode_cuda_graph_metadata     = {}
    backend.target_verify_metadata         = {}

    return backend


def _make_trtllm_backend(
    config: BenchmarkConfig,
    max_kv_len: int,
    req_to_token: torch.Tensor,
    device: torch.device,
):
    """
    Construct TRTLLMHAAttnBackend by setting only the attributes read by
    init_forward_metadata + forward_decode/forward_extend.

    Bypasses FlashInferAttnBackend.__init__ which pulls in ~20 model_runner
    attributes irrelevant to the TRTLLM kernel path.
    """
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)

    backend.workspace_buffer             = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    backend.page_size                    = config.block_size
    backend.data_type                    = config.dtype   # kv cache dtype
    backend.q_data_type                  = config.dtype   # query dtype
    backend.max_context_len              = max_kv_len
    backend.req_to_token                 = req_to_token   # used by some paths
    backend.device                       = device
    backend.topk                         = 0
    backend.speculative_step_id          = 0
    backend.speculative_num_draft_tokens = 0
    backend.use_sliding_window_kv_pool   = False
    backend._swa_kv_pool                 = None
    backend.forward_metadata             = None
    backend.target_verify_metadata       = {}
    backend.decode_cuda_graph_metadata   = {}

    return backend


def _make_layer(config: BenchmarkConfig, layer_id: int = 0):
    """
    Mock RadixAttention with the attributes read by forward_decode/forward_extend
    on a standard fp16 non-SWA decoder-only layer.
    """
    from sglang.srt.layers.radix_attention import AttentionType

    return SimpleNamespace(
        layer_id         = layer_id,
        tp_q_head_num    = config.num_q_heads,
        tp_k_head_num    = config.num_kv_heads,
        tp_v_head_num    = config.num_kv_heads,
        head_dim         = config.head_dim,
        v_head_dim       = config.head_dim,
        scaling          = get_attention_scale(config.head_dim),
        sliding_window_size = -1,
        is_cross_attention  = False,
        logit_cap           = 0.0,
        k_scale             = None,
        v_scale             = None,
        attn_type           = AttentionType.DECODER,
    )


def _make_forward_batch(
    q_lens: list[int],
    kv_lens: list[int],
    req_to_token: torch.Tensor,
    kv_pool: _MockKVPool,
    device: torch.device,
    is_decode: bool,
):
    """
    Build a minimal ForwardBatch namespace covering both the decode and extend
    (prefill / spec-decode) paths of init_forward_metadata.

    extend_prefix_lens_cpu: kv_len - q_len per request.
      = 0   for pure prefill (q_len == kv_len)         → any(...) is False
      > 0   for spec-decode / chunked extend            → any(...) is True
      In the latter case init_forward_metadata uses extend_seq_lens for cu_seqlens_q.
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    batch_size = len(kv_lens)

    return SimpleNamespace(
        batch_size         = batch_size,
        seq_lens           = torch.tensor(kv_lens, dtype=torch.int32, device=device),
        seq_lens_cpu       = torch.tensor(kv_lens, dtype=torch.int32),
        req_pool_indices   = torch.arange(batch_size, dtype=torch.int32, device=device),
        req_to_token_pool  = SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool   = kv_pool,
        out_cache_loc      = None,    # only needed when save_kv_cache=True
        spec_info          = None,    # no speculative decoding
        forward_mode       = ForwardMode.DECODE if is_decode else ForwardMode.EXTEND,
        attn_attend_prefix_cache = None,
        encoder_lens       = None,    # decoder-only model
        # Extend-path metadata (used when any(extend_prefix_lens_cpu) is True)
        extend_prefix_lens_cpu = [kv_len - q_len for kv_len, q_len in zip(kv_lens, q_lens)],
        extend_seq_lens        = torch.tensor(q_lens, dtype=torch.int32, device=device),
        extend_seq_lens_cpu    = q_lens,
    )


# ---------------------------------------------------------------------------
# Benchmark execution  (mirrors vllm/benchmarks/attention_benchmarks/runner.py)
# ---------------------------------------------------------------------------

def _run_single_benchmark(
    call_all_layers,
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[list[float], dict]:
    """Warmup, optional CUDA graph capture, timed loop."""
    for _ in range(config.warmup_iters):
        call_all_layers()
    torch.cuda.synchronize()

    if config.use_cuda_graphs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call_all_layers()
        benchmark_fn = graph.replay
    else:
        benchmark_fn = call_all_layers

    # Delimit the timed window for nsys --capture-range=cudaProfilerApi.
    # No-op when nsys is not attached.
    torch.cuda.cudart().cudaProfilerStart()

    times = []
    for i in range(config.repeats):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        benchmark_fn()
        end.record()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        times.append(start.elapsed_time(end) / 1000.0 / config.num_layers)

    torch.cuda.cudart().cudaProfilerStop()

    mem_stats = {}
    if config.profile_memory:
        mem_stats = {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb":  torch.cuda.memory_reserved(device)  / 1024**2,
        }

    return times, mem_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_attention_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run a direct-backend attention benchmark for SGLang FA3, FA4, or TRTLLM-MHA.

    Uses FlashAttentionBackend / TRTLLMHAAttnBackend directly so the exact
    production init_forward_metadata + forward_decode/forward_extend code paths
    are exercised.  save_kv_cache=False excludes the KV write from timing,
    consistent with vLLM's benchmark methodology.

    Args:
        config: config.backend must be "fa3", "fa4", or "trtllm".
    """
    device    = torch.device(config.device)
    dtype     = config.dtype

    requests  = parse_batch_spec(config.batch_spec)
    q_lens    = [r.q_len  for r in requests]
    kv_lens   = [r.kv_len for r in requests]
    total_q   = sum(q_lens)
    batch_size = len(q_lens)
    is_decode  = max(q_lens) == 1

    max_kv            = max(kv_lens)
    max_blocks_per_req = math.ceil(max_kv / config.block_size)
    num_blocks_total   = batch_size * max_blocks_per_req
    num_slots          = num_blocks_total * config.block_size

    # --- Build sglang-native data structures ---
    req_to_token = _build_req_to_token(kv_lens, config.block_size, device)

    k_buffers, v_buffers = _create_flat_kv_buffers(config, num_slots, device, dtype)
    kv_pool = _MockKVPool(k_buffers, v_buffers)

    q_list = _create_input_tensors(config, total_q, device, dtype)
    layers  = [_make_layer(config, layer_id=i) for i in range(config.num_layers)]

    forward_batch = _make_forward_batch(q_lens, kv_lens, req_to_token, kv_pool, device, is_decode)

    # --- Construct backend and initialise metadata ---
    backend_name = config.backend.lower()

    if backend_name in ("fa3", "fa4"):
        backend = _make_fa_backend(config, device)

    elif backend_name == "trtllm":
        backend = _make_trtllm_backend(config, max_kv, req_to_token, device)

    else:
        raise ValueError(f"Unknown backend: '{config.backend}'. Valid: fa3, fa4, trtllm")

    backend.init_forward_metadata(forward_batch)

    # Workaround: Call flash_attn_with_kvcache_fa4 directly for both decode
    # and extend paths because:
    #   1. forward_decode uses module-level flash_attn_with_kvcache (always FA3)
    #      regardless of fa_impl_ver — the FA3 AOT kernel crashes on SM100.
    #   2. init_forward_metadata stores the raw req_to_token slice as page_table
    #      (shape [batch, max_seq_len_k], values are slot indices). FA4 expects a
    #      block-level page table (shape [batch, max_blocks_per_req], values are
    #      block indices). Both decode and extend metadata have this mismatch.
    if backend_name == "fa4":
        from sglang.jit_kernel.flash_attention_v4 import (
            flash_attn_varlen_func as _fa4_varlen,
        )
        _scale = get_attention_scale(config.head_dim)

        # Convert slot-index page_table to block-index page_table
        meta = backend.forward_metadata
        strided = torch.arange(
            0, meta.max_seq_len_k, config.block_size, device=device
        )
        _block_page_table = meta.page_table[:, strided] // config.block_size

        # Pre-view KV buffers outside the timed loop
        _kv_pairs = []
        for i in range(config.num_layers):
            kv_k, kv_v = kv_pool.get_kv_buffer(i)
            _kv_pairs.append((
                kv_k.view(-1, config.block_size, config.num_kv_heads, config.head_dim),
                kv_v.view(-1, config.block_size, config.num_kv_heads, config.head_dim),
            ))

        _num_splits = backend.num_splits if backend.num_splits != 0 else 1

        def call_all_layers():
            for (q, layer), (kv_k, kv_v) in zip(zip(q_list, layers), _kv_pairs):
                _fa4_varlen(
                    q=q.view(-1, config.num_q_heads, config.head_dim),
                    k=kv_k,
                    v=kv_v,
                    cu_seqlens_q=meta.cu_seqlens_q,
                    seqused_k=meta.cache_seqlens_int32,
                    max_seqlen_q=meta.max_seq_len_q,
                    max_seqlen_k=meta.max_seq_len_k,
                    page_table=_block_page_table,
                    softmax_scale=_scale,
                    causal=True,
                    num_splits=_num_splits,
                )
    else:
        forward_fn = backend.forward_decode if is_decode else backend.forward_extend

        def call_all_layers():
            for i, q in enumerate(q_list):
                forward_fn(
                    q=q,
                    k=None,
                    v=None,
                    layer=layers[i],
                    forward_batch=forward_batch,
                    save_kv_cache=False,   # exclude KV write from timing
                )

    times, mem_stats = _run_single_benchmark(call_all_layers, config, device)

    mean_time  = float(np.mean(times))
    throughput = total_q / mean_time if mean_time > 0 else 0.0

    return BenchmarkResult(
        config=config,
        mean_time=mean_time,
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=mem_stats.get("allocated_mb"),
        memory_reserved_mb=mem_stats.get("reserved_mb"),
    )
