#!/usr/bin/env python3
"""
SGLang Attention Benchmark

Benchmark FA4 and TRTLLM-MHA attention backends with the batch-spec grammar.
Calls kernels directly — no model loading, no scheduler overhead.

Examples:
    # Compare both backends
    python benchmark.py --config configs/standard_attention.yaml

    # Quick CLI run
    python benchmark.py --backends fa4 trtllm --batch-specs "q2k" "8q1s1k"

    # Parameter sweep (CLI)
    python benchmark.py --backend fa4 --batch-specs "64q1s1k" \\
                        --sweep-param num_q_heads --sweep-values 32 40 64
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from batch_spec import parse_batch_spec
from common import (
    BenchmarkConfig,
    BenchmarkResult,
    ModelParameterSweep,
    ParameterSweep,
    ResultsFormatter,
    batch_spec_sort_key,
)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark, returning an error result on failure."""
    try:
        from runner import run_attention_benchmark
        return run_attention_benchmark(config)
    except Exception as e:
        return BenchmarkResult(
            config=config,
            mean_time=float("inf"),
            std_time=0,
            min_time=float("inf"),
            max_time=float("inf"),
            error=str(e),
        )


def run_model_parameter_sweep(
    backends: list[str],
    batch_specs: list[str],
    base_config_args: dict,
    sweep: ModelParameterSweep,
    console: Console,
) -> list[BenchmarkResult]:
    all_results = []
    console.print(
        f"[yellow]Model sweep mode: testing {sweep.param_name} = {sweep.values}[/]"
    )
    total = len(backends) * len(batch_specs) * len(sweep.values)

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for backend in backends:
            for spec in batch_specs:
                for value in sweep.values:
                    config_args = base_config_args.copy()
                    config_args[sweep.param_name] = value
                    clean_config = BenchmarkConfig(
                        backend=backend, batch_spec=spec, **config_args
                    )
                    result = run_benchmark(clean_config)
                    backend_label = sweep.get_label(backend, value)
                    labeled_config = replace(result.config, backend=backend_label)
                    result = replace(result, config=labeled_config)
                    all_results.append(result)
                    if not result.success:
                        console.print(
                            f"[red]Error {backend} {spec} {sweep.param_name}="
                            f"{value}: {result.error}[/]"
                        )
                    pbar.update(1)

    console.print("\n[bold green]Model Parameter Sweep Results:[/]")
    formatter = ResultsFormatter(console)

    by_param_value = {}
    backend_mapping = {}

    for r in all_results:
        labeled_backend = r.config.backend
        for backend in backends:
            for value in sweep.values:
                expected_label = sweep.get_label(backend, value)
                if labeled_backend == expected_label:
                    backend_mapping[labeled_backend] = backend
                    param_value = str(value)
                    if param_value not in by_param_value:
                        by_param_value[param_value] = []
                    by_param_value[param_value].append(r)
                    break

    sorted_param_values = sorted(
        by_param_value.keys(), key=lambda x: int(x) if x.isdigit() else x
    )

    for param_value in sorted_param_values:
        console.print(f"\n[bold cyan]{sweep.param_name} = {param_value}[/]")
        param_results = by_param_value[param_value]
        modified_results = []
        for r in param_results:
            original_backend = backend_mapping[r.config.backend]
            modified_config = replace(r.config, backend=original_backend)
            modified_result = replace(r, config=modified_config)
            modified_results.append(modified_result)
        formatter.print_table(modified_results, backends, compare_to_fastest=True)

    return all_results


def run_parameter_sweep(
    backends: list[str],
    batch_specs: list[str],
    base_config_args: dict,
    sweep: ParameterSweep,
    console: Console,
) -> list[BenchmarkResult]:
    all_results = []

    sweep_values = list(sweep.values)
    if sweep.include_auto:
        sweep_values.append("auto")

    console.print(f"[yellow]Sweep mode: testing {sweep.param_name} = {sweep_values}[/]")
    total = len(backends) * len(batch_specs) * len(sweep_values)

    with tqdm(total=total, desc="Benchmarking") as pbar:
        for backend in backends:
            for spec in batch_specs:
                for value in sweep_values:
                    config = BenchmarkConfig(
                        backend=backend, batch_spec=spec, **base_config_args
                    )
                    result = run_benchmark(config)
                    backend_label = sweep.get_label(backend, value)
                    labeled_config = replace(result.config, backend=backend_label)
                    result = replace(result, config=labeled_config)
                    all_results.append(result)
                    if not result.success:
                        console.print(
                            f"[red]Error {backend} {spec} {sweep.param_name}="
                            f"{value}: {result.error}[/]"
                        )
                    pbar.update(1)

    console.print("\n[bold green]Sweep Results:[/]")
    backend_labels = [sweep.get_label(b, v) for b in backends for v in sweep_values]
    formatter = ResultsFormatter(console)
    formatter.print_table(all_results, backend_labels)

    console.print(f"\n[bold cyan]Optimal {sweep.param_name} per batch spec:[/]")
    by_spec = {}
    for r in all_results:
        if r.success:
            spec = r.config.batch_spec
            if spec not in by_spec:
                by_spec[spec] = []
            by_spec[spec].append(r)

    for spec in sorted(by_spec.keys(), key=batch_spec_sort_key):
        results = by_spec[spec]
        best = min(results, key=lambda r: r.mean_time)
        console.print(
            f"  {spec}: [bold green]{best.config.backend}[/] ({best.mean_time:.6f}s)"
        )

    return all_results


def load_config_from_yaml(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_batch_specs_from_ranges(ranges: list[dict]) -> list[str]:
    """Generate batch specs from range specifications (see YAML examples)."""
    import itertools

    all_specs = []
    for range_spec in ranges:
        template = range_spec.get("template")
        if not template:
            raise ValueError("Range specification must include 'template'")

        range_params = {}
        for key, value in range_spec.items():
            if key == "template":
                continue
            if isinstance(value, dict) and "start" in value:
                start = value["start"]
                stop = value["stop"]
                step = value.get("step", 1)
                end_inclusive = value.get("end_inclusive", True)
                if end_inclusive:
                    range_params[key] = list(range(start, stop + 1, step))
                else:
                    range_params[key] = list(range(start, stop, step))
            else:
                range_params[key] = [value]

        if range_params:
            param_names = list(range_params.keys())
            param_values = [range_params[name] for name in param_names]
            for values in itertools.product(*param_values):
                params = dict(zip(param_names, values))
                spec = template.format(**params)
                all_specs.append(spec)
        else:
            all_specs.append(template)

    return all_specs


def main():
    parser = argparse.ArgumentParser(
        description="SGLang attention benchmark (FA4 vs TRTLLM-MHA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", help="Path to YAML config file")

    parser.add_argument("--backends", nargs="+", help="Backends to benchmark (fa4, trtllm)")
    parser.add_argument("--backend", help="Single backend")

    parser.add_argument("--batch-specs", nargs="+", default=None)

    parser.add_argument("--num-layers", type=int, default=10)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=16)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype", default="auto", choices=["auto", "fp8"]
    )
    parser.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA graphs to eliminate CPU overhead (default: True)",
    )

    parser.add_argument("--sweep-param", help="Parameter name to sweep")
    parser.add_argument("--sweep-values", type=int, nargs="+")

    parser.add_argument("--output-csv", help="Save results to CSV")
    parser.add_argument("--output-json", help="Save results to JSON")

    args = parser.parse_args()

    console = Console()
    console.print("[bold cyan]SGLang Attention Benchmark[/]")

    # -----------------------------------------------------------------------
    # YAML config loading (CLI args take precedence)
    # -----------------------------------------------------------------------
    if args.config:
        console.print(f"[yellow]Loading config from: {args.config}[/]")
        yaml_config = load_config_from_yaml(args.config)

        if "description" in yaml_config:
            console.print(f"[dim]{yaml_config['description']}[/]")

        cli_backends_provided = args.backend is not None or args.backends is not None
        if not cli_backends_provided:
            if "backend" in yaml_config:
                args.backend = yaml_config["backend"]
                args.backends = None
            elif "backends" in yaml_config:
                args.backends = yaml_config["backends"]
                args.backend = None
            elif "decode_backends" in yaml_config:
                args.backends = yaml_config["decode_backends"]
                args.backend = None

        args.mode = yaml_config.get("mode", None)

        cli_batch_specs_provided = args.batch_specs is not None
        if not cli_batch_specs_provided:
            if "batch_spec_ranges" in yaml_config:
                generated_specs = generate_batch_specs_from_ranges(
                    yaml_config["batch_spec_ranges"]
                )
                if "batch_specs" in yaml_config:
                    args.batch_specs = yaml_config["batch_specs"] + generated_specs
                else:
                    args.batch_specs = generated_specs
                console.print(
                    f"[dim]Generated {len(generated_specs)} batch specs from ranges[/]"
                )
            elif "batch_specs" in yaml_config:
                args.batch_specs = yaml_config["batch_specs"]

        args.batch_sizes = yaml_config.get("batch_sizes", None)

        if "model" in yaml_config:
            model = yaml_config["model"]
            args.num_layers = model.get("num_layers", args.num_layers)
            args.head_dim = model.get("head_dim", args.head_dim)
            args.num_q_heads = model.get("num_q_heads", args.num_q_heads)
            args.num_kv_heads = model.get("num_kv_heads", args.num_kv_heads)
            args.block_size = model.get("block_size", args.block_size)

        if "device" in yaml_config:
            args.device = yaml_config["device"]
        if "repeats" in yaml_config:
            args.repeats = yaml_config["repeats"]
        if "warmup_iters" in yaml_config:
            args.warmup_iters = yaml_config["warmup_iters"]
        if "profile_memory" in yaml_config:
            args.profile_memory = yaml_config["profile_memory"]
        if "kv_cache_dtype" in yaml_config:
            args.kv_cache_dtype = yaml_config["kv_cache_dtype"]
        if "cuda_graphs" in yaml_config:
            args.cuda_graphs = yaml_config["cuda_graphs"]

        if "parameter_sweep" in yaml_config:
            sweep_config = yaml_config["parameter_sweep"]
            args.parameter_sweep = ParameterSweep(
                param_name=sweep_config["param_name"],
                values=sweep_config["values"],
                include_auto=sweep_config.get("include_auto", False),
                label_format=sweep_config.get(
                    "label_format", "{backend}_{param_name}_{value}"
                ),
            )
        else:
            args.parameter_sweep = None

        if "model_parameter_sweep" in yaml_config:
            sweep_config = yaml_config["model_parameter_sweep"]
            args.model_parameter_sweep = ModelParameterSweep(
                param_name=sweep_config["param_name"],
                values=sweep_config["values"],
                label_format=sweep_config.get(
                    "label_format", "{backend}_{param_name}_{value}"
                ),
            )
        else:
            args.model_parameter_sweep = None

        if "output" in yaml_config:
            output = yaml_config["output"]
            if "csv" in output and not args.output_csv:
                args.output_csv = output["csv"]
            if "json" in output and not args.output_json:
                args.output_json = output["json"]

        console.print()

    # CLI-based parameter sweep
    if (
        (not hasattr(args, "parameter_sweep") or args.parameter_sweep is None)
        and args.sweep_param
        and args.sweep_values
    ):
        args.parameter_sweep = ParameterSweep(
            param_name=args.sweep_param,
            values=args.sweep_values,
            include_auto=False,
            label_format="{backend}_{param_name}_{value}",
        )

    backends = args.backends or ([args.backend] if args.backend else ["fa4"])
    if not args.batch_specs:
        args.batch_specs = ["q2k", "8q1s1k"]

    console.print(f"Backends: {', '.join(backends)}")
    console.print(f"Batch specs: {', '.join(args.batch_specs)}")
    console.print(f"KV cache dtype: {args.kv_cache_dtype}")
    console.print(f"CUDA graphs: {args.cuda_graphs}")
    console.print()

    all_results = []

    # -----------------------------------------------------------------------
    # Model parameter sweep
    # -----------------------------------------------------------------------
    if hasattr(args, "model_parameter_sweep") and args.model_parameter_sweep:
        base_config_args = {
            "num_layers": args.num_layers,
            "head_dim": args.head_dim,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "block_size": args.block_size,
            "device": args.device,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "profile_memory": args.profile_memory,
            "kv_cache_dtype": args.kv_cache_dtype,
            "use_cuda_graphs": args.cuda_graphs,
        }
        all_results = run_model_parameter_sweep(
            backends,
            args.batch_specs,
            base_config_args,
            args.model_parameter_sweep,
            console,
        )

    # -----------------------------------------------------------------------
    # Parameter sweep
    # -----------------------------------------------------------------------
    elif hasattr(args, "parameter_sweep") and args.parameter_sweep:
        base_config_args = {
            "num_layers": args.num_layers,
            "head_dim": args.head_dim,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "block_size": args.block_size,
            "device": args.device,
            "repeats": args.repeats,
            "warmup_iters": args.warmup_iters,
            "profile_memory": args.profile_memory,
            "kv_cache_dtype": args.kv_cache_dtype,
            "use_cuda_graphs": args.cuda_graphs,
        }
        all_results = run_parameter_sweep(
            backends, args.batch_specs, base_config_args, args.parameter_sweep, console
        )

    # -----------------------------------------------------------------------
    # Normal mode: compare backends across batch specs
    # -----------------------------------------------------------------------
    else:
        total = len(backends) * len(args.batch_specs)

        with tqdm(total=total, desc="Benchmarking") as pbar:
            for spec in args.batch_specs:
                for backend in backends:
                    config = BenchmarkConfig(
                        backend=backend,
                        batch_spec=spec,
                        num_layers=args.num_layers,
                        head_dim=args.head_dim,
                        num_q_heads=args.num_q_heads,
                        num_kv_heads=args.num_kv_heads,
                        block_size=args.block_size,
                        device=args.device,
                        repeats=args.repeats,
                        warmup_iters=args.warmup_iters,
                        profile_memory=args.profile_memory,
                        kv_cache_dtype=args.kv_cache_dtype,
                        use_cuda_graphs=args.cuda_graphs,
                    )

                    result = run_benchmark(config)
                    all_results.append(result)

                    if not result.success:
                        console.print(
                            f"[red]Error {backend} {spec}: {result.error}[/]"
                        )

                    pbar.update(1)

        console.print("\n[bold green]Results:[/]")
        formatter = ResultsFormatter(console)
        formatter.print_table(all_results, backends)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    if all_results:
        formatter = ResultsFormatter(console)
        if args.output_csv:
            formatter.save_csv(all_results, args.output_csv)
        if args.output_json:
            formatter.save_json(all_results, args.output_json)


if __name__ == "__main__":
    main()
