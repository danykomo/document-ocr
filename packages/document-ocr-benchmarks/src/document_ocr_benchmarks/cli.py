"""Command-line interface for the benchmark harness.

    document-ocr-bench gen-samples   # build synthetic dataset
    document-ocr-bench providers      # list candidates, capabilities, license
    document-ocr-bench validate       # check a dataset is well-formed
    document-ocr-bench run            # run the benchmark
    document-ocr-bench report         # build report from a results run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import get_settings
from .models import BenchmarkResult, CaptureCondition, DocumentType
from .providers.registry import PROVIDER_CLASSES, build_provider

console = Console()

DEFAULT_DATASET = "benchmarks/document-ocr"
DEFAULT_RESULTS = "benchmarks/document-ocr/results"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="document-ocr-bench")
def main() -> None:
    """Benchmark harness for Nigerian document OCR/VLM providers."""


# --------------------------------------------------------------------------- #
@main.command("gen-samples")
@click.option("--out", default=DEFAULT_DATASET, help="Dataset root directory.")
@click.option("--per-type", default=1, show_default=True, help="Base docs per document type.")
@click.option("--doc-types", default="", help="Comma-separated document types (default: all).")
@click.option("--conditions", default="", help="Comma-separated capture conditions (default set).")
@click.option("--seed", default=1000, show_default=True)
def gen_samples(out: str, per_type: int, doc_types: str, conditions: str, seed: int) -> None:
    """Generate synthetic Nigerian specimen documents + ground truth."""
    from .synth.generator import generate_dataset, DEFAULT_CONDITIONS, DEFAULT_DOC_TYPES

    dts = ([DocumentType(x.strip()) for x in doc_types.split(",") if x.strip()]
           or DEFAULT_DOC_TYPES)
    conds = ([CaptureCondition(x.strip()) for x in conditions.split(",") if x.strip()]
             or DEFAULT_CONDITIONS)
    manifest = generate_dataset(
        Path(out), doc_types=dts, conditions=conds, per_type=per_type, seed=seed,
    )
    console.print(
        f"[green]Generated {len(manifest.samples)} samples[/green] under {out}/samples, "
        f"manifest: {out}/manifests/{manifest.name}.json"
    )


# --------------------------------------------------------------------------- #
@main.command("providers")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
def providers_cmd(as_json: bool) -> None:
    """List candidate providers, capabilities, availability, and license posture."""
    settings = get_settings()
    reports = []
    for name in PROVIDER_CLASSES:
        try:
            runner = build_provider(name, settings)
            reports.append(runner.capability_report())
        except Exception as exc:  # pragma: no cover - defensive
            console.print(f"[red]{name}: {exc}[/red]")

    if as_json:
        click.echo(json.dumps([r.model_dump(mode="json") for r in reports], indent=2))
        return

    table = Table(title="OCR/VLM Provider Candidates")
    table.add_column("Provider"); table.add_column("Avail"); table.add_column("Caps")
    table.add_column("License"); table.add_column("Status"); table.add_column("Cost")
    table.add_column("On-prem")
    for r in reports:
        table.add_row(
            r.provider,
            "[green]yes[/green]" if r.available else f"[yellow]no[/yellow]",
            ",".join(c.value for c in r.capabilities),
            r.license_name or "—",
            r.license_status.value,
            r.cost_estimate.value,
            "yes" if r.on_prem else "no",
        )
    console.print(table)
    for r in reports:
        if not r.available and r.reason:
            console.print(f"  [dim]{r.provider}: {r.reason}[/dim]")


# --------------------------------------------------------------------------- #
@main.command("validate")
@click.option("--dataset", default=DEFAULT_DATASET, show_default=True)
@click.option("--manifest", "manifest_path", default="", help="Defaults to first manifest found.")
def validate_cmd(dataset: str, manifest_path: str) -> None:
    """Validate a dataset (images + expected files present, no duplicate ids)."""
    from .manifest import load_manifest, validate_dataset

    mpath = Path(manifest_path) if manifest_path else _find_manifest(dataset)
    if mpath is None:
        raise click.ClickException(f"No manifest found under {dataset}/manifests")
    manifest = load_manifest(mpath)
    problems = validate_dataset(manifest, dataset)
    if problems:
        console.print(f"[red]{len(problems)} problem(s):[/red]")
        for p in problems:
            console.print(f"  - {p}")
        raise SystemExit(1)
    console.print(f"[green]OK[/green] — {len(manifest.samples)} samples valid in {mpath}")


# --------------------------------------------------------------------------- #
@main.command("run")
@click.option("--dataset", default=DEFAULT_DATASET, show_default=True)
@click.option("--manifest", "manifest_path", default="", help="Defaults to first manifest found.")
@click.option("--providers", "providers", default="", help="Comma-separated (default: configured).")
@click.option("--results", default=DEFAULT_RESULTS, show_default=True)
@click.option("--no-text", is_flag=True, help="Skip raw text extraction.")
@click.option("--no-portrait", is_flag=True, help="Skip portrait extraction.")
@click.option("--no-classify", is_flag=True, help="Skip classification.")
@click.option("--doc-type", default="", help="Only run samples of this document type.")
@click.option("--report/--no-report", default=True, help="Build report after the run.")
def run_cmd(dataset, manifest_path, providers, results, no_text, no_portrait,
            no_classify, doc_type, report) -> None:
    """Run the benchmark across providers and samples."""
    from .harness import run_benchmark
    from .report import write_report

    settings = get_settings()
    mpath = Path(manifest_path) if manifest_path else _find_manifest(dataset)
    if mpath is None:
        raise click.ClickException(f"No manifest found under {dataset}/manifests")

    provider_list = ([p.strip() for p in providers.split(",") if p.strip()]
                     or settings.allowed_providers)

    sample_filter = None
    if doc_type:
        prefix = DocumentType(doc_type).value
        sample_filter = lambda sid: sid.startswith(prefix)  # noqa: E731

    run = run_benchmark(
        manifest_path=mpath, dataset_root=dataset, providers=provider_list,
        results_root=results, settings=settings,
        do_text=not no_text, do_portrait=not no_portrait, do_classify=not no_classify,
        sample_filter=sample_filter,
    )

    if report:
        out_dir = Path(results) / run.run_id
        json_path, md_path = write_report(run.results, out_dir, run.meta)
        console.print(f"[green]Report:[/green] {md_path}")
        _print_quick_ranking(run.results)


# --------------------------------------------------------------------------- #
@main.command("report")
@click.argument("results_dir", type=click.Path(exists=True))
def report_cmd(results_dir: str) -> None:
    """(Re)build the report from a results directory containing results.json."""
    from .report import write_report

    rdir = Path(results_dir)
    results_file = rdir / "results.json"
    if not results_file.exists():
        raise click.ClickException(f"No results.json in {rdir}")
    rows = [BenchmarkResult.model_validate(r) for r in json.loads(results_file.read_text())]
    meta = json.loads((rdir / "run_meta.json").read_text()) if (rdir / "run_meta.json").exists() else None
    json_path, md_path = write_report(rows, rdir, meta)
    console.print(f"[green]Report written:[/green] {md_path}")
    _print_quick_ranking(rows)


# --------------------------------------------------------------------------- #
def _find_manifest(dataset: str) -> Optional[Path]:
    mdir = Path(dataset) / "manifests"
    if not mdir.exists():
        return None
    manifests = sorted(mdir.glob("*.json"))
    return manifests[0] if manifests else None


def _print_quick_ranking(rows: list[BenchmarkResult]) -> None:
    from .report import build_summary

    summary = build_summary(rows)
    table = Table(title="Composite ranking")
    table.add_column("Candidate"); table.add_column("Composite", justify="right")
    table.add_column("Field acc", justify="right"); table.add_column("p50 ms", justify="right")
    for c in summary["candidates"]:
        table.add_row(c["candidate"], f"{c['composite_score']:.3f}",
                      f"{c['field_accuracy']:.3f}", f"{c['latency_p50_ms']:.0f}")
    console.print(table)
    console.print(f"[bold]Recommended default:[/bold] {summary['recommendation']['default_provider']}")


if __name__ == "__main__":
    main()
