"""Common utilities for attention benchmarking."""

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from batch_spec import get_batch_type, parse_batch_spec
from rich.console import Console
from rich.table import Table


def batch_spec_sort_key(spec: str) -> tuple[int, int, int]:
    """
    Extract sorting key from batch spec: (batch_size, max_q_len, max_kv_len).

    This ensures results are sorted by batch size first, then query length,
    then sequence length, rather than alphabetically.
    """
    try:
        requests = parse_batch_spec(spec)
        batch_size = len(requests)
        max_q_len = max(r.q_len for r in requests) if requests else 0
        max_kv_len = max(r.kv_len for r in requests) if requests else 0
        return (batch_size, max_q_len, max_kv_len)
    except Exception:
        return (0, 0, 0)


@dataclass
class ParameterSweep:
    """Configuration for sweeping a backend parameter."""

    param_name: str
    values: list[Any]
    include_auto: bool = False
    label_format: str = "{backend}_{param_name}_{value}"

    def get_label(self, backend: str, value: Any) -> str:
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


@dataclass
class ModelParameterSweep:
    """Configuration for sweeping a model configuration parameter."""

    param_name: str
    values: list[Any]
    label_format: str = "{backend}_{param_name}_{value}"

    def get_label(self, backend: str, value: Any) -> str:
        return self.label_format.format(
            backend=backend, param_name=self.param_name, value=value
        )


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    backend: str
    batch_spec: str
    num_layers: int
    head_dim: int
    num_q_heads: int
    num_kv_heads: int
    block_size: int
    device: str
    dtype: torch.dtype = torch.float16
    repeats: int = 1
    warmup_iters: int = 3
    profile_memory: bool = False
    use_cuda_graphs: bool = False

    # "auto" or "fp8"
    kv_cache_dtype: str = "auto"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    mean_time: float  # seconds
    std_time: float  # seconds
    min_time: float  # seconds
    max_time: float  # seconds
    throughput_tokens_per_sec: float | None = None
    memory_allocated_mb: float | None = None
    memory_reserved_mb: float | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "error": self.error,
        }


class ResultsFormatter:
    """Format and display benchmark results."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_table(
        self,
        results: list[BenchmarkResult],
        backends: list[str],
        compare_to_fastest: bool = True,
    ):
        # Group by batch spec
        by_spec = {}
        specs_order = []
        for r in results:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = {}
                specs_order.append(spec)
            by_spec[spec][r.config.backend] = r

        specs_order = sorted(by_spec.keys(), key=batch_spec_sort_key)

        def shorten_backend_name(name: str) -> str:
            return name

        table = Table(title="Attention Benchmark Results")
        table.add_column("Batch\nSpec", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Batch\nSize", justify="right", no_wrap=True)

        multi = len(backends) > 1
        for backend in backends:
            short_name = shorten_backend_name(backend)
            table.add_column(f"{short_name}\nTime (s)", justify="right", no_wrap=False)
            table.add_column(f"{short_name}\nTok/s", justify="right", no_wrap=False)
            if multi and compare_to_fastest:
                table.add_column(f"{short_name}\nvs Best", justify="right", no_wrap=False)

        for spec in specs_order:
            spec_results = by_spec[spec]
            times = {b: r.mean_time for b, r in spec_results.items() if r.success}
            best_time = min(times.values()) if times else 0.0

            batch_type = get_batch_type(spec)
            batch_size = len(parse_batch_spec(spec))
            row = [spec, batch_type, str(batch_size)]
            for backend in backends:
                if backend in spec_results:
                    r = spec_results[backend]
                    if r.success:
                        row.append(f"{r.mean_time:.6f}")
                        tps = r.throughput_tokens_per_sec
                        row.append(f"{tps:.0f}" if tps is not None else "-")
                        if multi and compare_to_fastest:
                            pct = (r.mean_time / best_time * 100) if best_time > 0 else 0
                            pct_str = f"{pct:.1f}%"
                            if r.mean_time == best_time:
                                pct_str = f"[bold green]{pct_str}[/]"
                            row.append(pct_str)
                    else:
                        row.append("[red]ERROR[/]")
                        row.append("-")
                        if multi and compare_to_fastest:
                            row.append("-")
                else:
                    row.append("-")
                    row.append("-")
                    if multi and compare_to_fastest:
                        row.append("-")

            table.add_row(*row)

        self.console.print(table)

    def save_csv(self, results: list[BenchmarkResult], path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "backend",
                    "batch_spec",
                    "num_layers",
                    "kv_cache_dtype",
                    "mean_time",
                    "std_time",
                    "throughput",
                    "memory_mb",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "backend": r.config.backend,
                        "batch_spec": r.config.batch_spec,
                        "num_layers": r.config.num_layers,
                        "kv_cache_dtype": r.config.kv_cache_dtype,
                        "mean_time": r.mean_time,
                        "std_time": r.std_time,
                        "throughput": r.throughput_tokens_per_sec or 0,
                        "memory_mb": r.memory_allocated_mb or 0,
                    }
                )

        self.console.print(f"[green]Saved CSV results to {path}[/]")

    def save_json(self, results: list[BenchmarkResult], path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.console.print(f"[green]Saved JSON results to {path}[/]")


def get_attention_scale(head_dim: int) -> float:
    """Compute attention scale factor (1/sqrt(d))."""
    return 1.0 / math.sqrt(head_dim)
