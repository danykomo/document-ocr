"""Benchmark orchestrator.

Runs each configured provider over each sample, measuring latency and resources,
scoring against ground truth, and writing structured evidence to disk. Provider
or per-sample failures are caught and recorded so one bad case never aborts the
run (spec: "failure behavior").
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from .config import Settings, get_settings
from .imaging import LoadedImage
from .manifest import load_expected, load_manifest, resolve_image_path
from .models import BenchmarkResult, DocumentType, Manifest
from .providers.base import ProviderRunner
from .providers.registry import build_provider
from .resources import gpu_memory_used_mb, track_peak_memory
from .scoring import score_sample
from .schemas import schema_for

console = Console()


class BenchmarkRun:
    """Holds the results and metadata of a single benchmark invocation."""

    def __init__(self, run_id: str, results: list[BenchmarkResult], meta: dict):
        self.run_id = run_id
        self.results = results
        self.meta = meta


def run_benchmark(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    providers: list[str],
    results_root: str | Path,
    settings: Optional[Settings] = None,
    do_classify: bool = True,
    do_text: bool = True,
    do_fields: bool = True,
    do_portrait: bool = True,
    sample_filter: Optional[Callable[[str], bool]] = None,
    progress: bool = True,
) -> BenchmarkRun:
    settings = settings or get_settings()
    manifest: Manifest = load_manifest(manifest_path)
    dataset_root = Path(dataset_root)
    expected_dir = dataset_root / "expected"

    samples = [s for s in manifest.samples if (sample_filter is None or sample_filter(s.sample_id))]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(results_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate providers up-front; record availability.
    runners: dict[str, ProviderRunner] = {}
    availability: dict[str, dict] = {}
    for name in providers:
        try:
            runner = build_provider(name, settings)
        except KeyError as exc:
            availability[name] = {"available": False, "reason": str(exc)}
            continue
        ok, reason = runner.is_available()
        availability[name] = {"available": ok, "reason": reason}
        if ok:
            runners[name] = runner
        else:
            console.print(f"[yellow]Skipping {name}: {reason}[/yellow]")

    results: list[BenchmarkResult] = []
    jsonl = (out_dir / "results.jsonl").open("w")

    total = len(runners) * len(samples)
    prog_ctx = (
        Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), MofNCompleteColumn(), console=console,
        )
        if progress and total
        else None
    )
    task_id = None
    if prog_ctx:
        prog_ctx.start()
        task_id = prog_ctx.add_task("Benchmarking", total=total)

    try:
        for name, runner in runners.items():
            for sample in samples:
                if prog_ctx:
                    prog_ctx.update(task_id, description=f"{name} · {sample.sample_id}")
                result = _run_one(runner, sample, dataset_root, expected_dir,
                                  do_classify, do_text, do_fields, do_portrait, settings)
                results.append(result)
                jsonl.write(json.dumps(result.model_dump(by_alias=True)) + "\n")
                if prog_ctx:
                    prog_ctx.advance(task_id)
    finally:
        jsonl.close()
        if prog_ctx:
            prog_ctx.stop()

    meta = {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "manifest_name": manifest.name,
        "dataset_root": str(dataset_root),
        "providers_requested": providers,
        "provider_availability": availability,
        "sample_count": len(samples),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ops": {
            "classify": do_classify, "text": do_text,
            "fields": do_fields, "portrait": do_portrait,
        },
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (out_dir / "results.json").write_text(
        json.dumps([r.model_dump(by_alias=True) for r in results], indent=2)
    )
    console.print(f"[green]Wrote {len(results)} results to {out_dir}[/green]")

    # Surface providers that errored, so a broken VLM isn't mistaken for a real
    # (fast, low-scoring) result. Quality/portrait run locally and can succeed
    # even when the model endpoint is failing, which otherwise looks "simulated".
    by_provider: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_provider.setdefault(r.candidate, []).append(r)
    for name, rows in by_provider.items():
        errored = [r for r in rows if r.error_code]
        if not errored:
            continue
        sample_err = next((r.error for r in errored if r.error), "")
        console.print(
            f"[yellow]⚠ {name}: {len(errored)}/{len(rows)} samples errored[/yellow] "
            f"(e.g. {sample_err[:160]})"
        )
        if len(errored) == len(rows):
            console.print(
                f"  [red]{name} failed on every sample — its scores are not real "
                f"inference. Check the model endpoint before trusting the ranking.[/red]"
            )
    return BenchmarkRun(run_id, results, meta)


def _run_one(
    runner: ProviderRunner,
    sample,
    dataset_root: Path,
    expected_dir: Path,
    do_classify: bool,
    do_text: bool,
    do_fields: bool,
    do_portrait: bool,
    settings: Settings,
) -> BenchmarkResult:
    expected = load_expected(expected_dir, sample.sample_id)
    schema = schema_for(sample.document_type)

    if expected is None:
        return BenchmarkResult(
            candidate=runner.name,
            document_type=sample.document_type,
            sample_id=sample.sample_id,
            error="missing expected ground truth",
            error_code="MISSING_EXPECTED",
            cost_estimate=runner.cost_estimate,
            license_status=runner.license_status,
            on_prem=runner.on_prem,
        )

    try:
        image = LoadedImage.from_path(resolve_image_path(dataset_root, sample))
    except Exception as exc:
        return BenchmarkResult(
            candidate=runner.name,
            document_type=sample.document_type,
            sample_id=sample.sample_id,
            capture_condition=sample.capture_condition,
            error=f"image load failed: {type(exc).__name__}",
            error_code="INVALID_IMAGE",
            cost_estimate=runner.cost_estimate,
            license_status=runner.license_status,
            on_prem=runner.on_prem,
        )

    gpu_before = gpu_memory_used_mb() if settings.gpu_query_enabled else None
    with track_peak_memory() as peak_mb:
        run = runner.analyze(
            image, schema,
            do_classify=do_classify, do_text=do_text,
            do_fields=do_fields, do_portrait=do_portrait,
        )
    gpu_after = gpu_memory_used_mb() if settings.gpu_query_enabled else None
    gpu_mb = max(filter(None, [gpu_before, gpu_after]), default=None)

    result = score_sample(
        provider=runner, run=run, expected=expected, schema=schema,
        gpu_memory_mb=gpu_mb, peak_memory_mb=peak_mb(),
    )
    result.capture_condition = sample.capture_condition
    return result
