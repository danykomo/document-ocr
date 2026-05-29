"""Aggregate benchmark results into a recommendation report.

Produces a JSON summary plus a Markdown report. The composite score weighs
accuracy alongside the operational dimensions the spec insists on (latency,
cost, license, schema-following, portrait), so "1% better OCR at 10x GPU cost"
does not automatically win. Per-operation winners are also reported, because the
spec allows different defaults for classify / extract_text / extract_fields /
extract_portrait.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .models import BenchmarkResult, CostEstimate, LicenseStatus

# Composite weights. Tunable; documented in the report so choices are auditable.
WEIGHTS = {
    "field_accuracy": 0.35,
    "required_recall": 0.15,
    "classification": 0.10,
    "schema_following": 0.10,
    "latency": 0.10,
    "portrait": 0.10,
    "cost": 0.05,
    "license": 0.05,
}
# Latency budget (ms) above which the latency sub-score hits zero.
LATENCY_BUDGET_MS = 8000.0
# When two candidates are within this composite margin, prefer the cheaper one.
COST_TIEBREAK_MARGIN = 0.03

_COST_SCORE = {CostEstimate.LOW: 1.0, CostEstimate.MEDIUM: 0.6, CostEstimate.HIGH: 0.3}
_LICENSE_SCORE = {
    LicenseStatus.APPROVED: 1.0,
    LicenseStatus.NEEDS_REVIEW: 0.5,
    LicenseStatus.REJECTED: 0.0,
}


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return round(s[f] + (s[c] - s[f]) * (k - f), 1)


@dataclass
class CandidateSummary:
    candidate: str
    n: int
    field_accuracy: float
    required_recall: float
    classification_accuracy: Optional[float]
    schema_following_rate: float
    portrait_success_rate: Optional[float]
    portrait_quality: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    gpu_memory_mb: Optional[float]
    cost_estimate: str
    license_status: str
    on_prem: bool
    error_rate: float
    composite_score: float = 0.0
    by_document_type: dict[str, float] = field(default_factory=dict)
    by_capture_condition: dict[str, float] = field(default_factory=dict)


def _summarize_candidate(name: str, rows: list[BenchmarkResult]) -> CandidateSummary:
    field_acc = [r.field_accuracy for r in rows]
    recall = [r.required_field_recall for r in rows]
    cls = [1.0 if r.classification_correct else 0.0
           for r in rows if r.classification_correct is not None]
    schema = [1.0 if r.schema_followed else 0.0 for r in rows]
    portrait_expected = [r for r in rows if r.portrait_expected]
    portrait_rate = (
        _mean([1.0 if r.portrait_extracted else 0.0 for r in portrait_expected])
        if portrait_expected else None
    )
    portrait_q = [r.portrait_quality_score for r in rows if r.portrait_extracted]
    latencies = [r.latency_ms for r in rows if r.latency_ms]
    gpus = [r.gpu_memory_mb for r in rows if r.gpu_memory_mb is not None]
    errors = [1.0 if r.error_code and r.error_code != "PROVIDER_PARTIAL_FAILURE" else 0.0
              for r in rows]

    by_doc: dict[str, list[float]] = {}
    by_cond: dict[str, list[float]] = {}
    for r in rows:
        by_doc.setdefault(r.document_type.value, []).append(r.field_accuracy)
        by_cond.setdefault(r.capture_condition.value, []).append(r.field_accuracy)

    return CandidateSummary(
        candidate=name,
        n=len(rows),
        field_accuracy=_mean(field_acc),
        required_recall=_mean(recall),
        classification_accuracy=_mean(cls) if cls else None,
        schema_following_rate=_mean(schema),
        portrait_success_rate=portrait_rate,
        portrait_quality=_mean(portrait_q),
        latency_p50_ms=_pct(latencies, 0.5),
        latency_p95_ms=_pct(latencies, 0.95),
        latency_mean_ms=_mean(latencies),
        gpu_memory_mb=round(_mean(gpus), 1) if gpus else None,
        cost_estimate=rows[0].cost_estimate.value,
        license_status=rows[0].license_status.value,
        on_prem=rows[0].on_prem,
        error_rate=_mean(errors),
        by_document_type={k: _mean(v) for k, v in sorted(by_doc.items())},
        by_capture_condition={k: _mean(v) for k, v in sorted(by_cond.items())},
    )


def _composite(s: CandidateSummary) -> float:
    latency_score = max(0.0, 1.0 - (s.latency_p50_ms / LATENCY_BUDGET_MS))
    cost_score = _COST_SCORE.get(CostEstimate(s.cost_estimate), 0.5)
    license_score = _LICENSE_SCORE.get(LicenseStatus(s.license_status), 0.5)
    score = (
        WEIGHTS["field_accuracy"] * s.field_accuracy
        + WEIGHTS["required_recall"] * s.required_recall
        + WEIGHTS["classification"] * (s.classification_accuracy or 0.0)
        + WEIGHTS["schema_following"] * s.schema_following_rate
        + WEIGHTS["latency"] * latency_score
        + WEIGHTS["portrait"] * (s.portrait_success_rate or 0.0)
        + WEIGHTS["cost"] * cost_score
        + WEIGHTS["license"] * license_score
    )
    return round(score, 4)


@dataclass
class Recommendation:
    default_provider: Optional[str]
    rationale: str
    per_operation: dict[str, Optional[str]]


def build_summary(results: list[BenchmarkResult]) -> dict:
    by_candidate: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_candidate.setdefault(r.candidate, []).append(r)

    summaries = [_summarize_candidate(name, rows) for name, rows in by_candidate.items()]
    for s in summaries:
        s.composite_score = _composite(s)
    summaries.sort(key=lambda s: s.composite_score, reverse=True)

    recommendation = _recommend(summaries)
    return {
        "candidates": [asdict(s) for s in summaries],
        "recommendation": asdict(recommendation),
        "weights": WEIGHTS,
        "latency_budget_ms": LATENCY_BUDGET_MS,
    }


def _best(summaries: list[CandidateSummary], key) -> Optional[str]:
    ranked = [s for s in summaries if key(s) is not None]
    if not ranked:
        return None
    return max(ranked, key=key).candidate


def _recommend(summaries: list[CandidateSummary]) -> Recommendation:
    if not summaries:
        return Recommendation(None, "No results.", {})

    top = summaries[0]
    default = top
    rationale = (
        f"{top.candidate} has the highest composite score ({top.composite_score})."
    )
    # Spec rule: prefer the cheaper/smaller model when accuracy is close enough.
    if len(summaries) > 1:
        runner_up = summaries[1]
        if (top.composite_score - runner_up.composite_score) <= COST_TIEBREAK_MARGIN:
            # Higher cost_score == cheaper to serve.
            cheaper = max([top, runner_up],
                          key=lambda s: _COST_SCORE.get(CostEstimate(s.cost_estimate), 0.5))
            if cheaper.candidate != top.candidate:
                default = cheaper
                rationale = (
                    f"{cheaper.candidate} is within {COST_TIEBREAK_MARGIN} composite of "
                    f"{top.candidate} ({cheaper.composite_score} vs {top.composite_score}) "
                    f"but cheaper to serve ({cheaper.cost_estimate} vs {top.cost_estimate}); "
                    "preferred per the on-prem cost rule."
                )

    per_op = {
        "classify": _best(summaries, lambda s: s.classification_accuracy),
        "extract_fields": _best(summaries, lambda s: s.field_accuracy),
        # No reference transcript yet, so text uses field accuracy as a proxy.
        "extract_text": _best(summaries, lambda s: s.field_accuracy),
        "extract_portrait": _best(
            summaries,
            lambda s: (s.portrait_success_rate or 0.0) * (s.portrait_quality or 0.0)
            if s.portrait_success_rate is not None else None,
        ),
    }
    return Recommendation(default.candidate, rationale, per_op)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def render_markdown(summary: dict, meta: Optional[dict] = None) -> str:
    cands = summary["candidates"]
    rec = summary["recommendation"]
    lines: list[str] = []
    lines.append("# Document OCR Provider Benchmark — Recommendation Report\n")
    if meta:
        lines.append(f"- Run: `{meta.get('run_id', 'n/a')}`")
        lines.append(f"- Manifest: `{meta.get('manifest_name', 'n/a')}` "
                     f"({meta.get('sample_count', '?')} samples)")
        lines.append(f"- Platform: {meta.get('platform', '?')} · Python {meta.get('python', '?')}")
        lines.append("")

    lines.append("## Recommendation\n")
    lines.append(f"**Default provider:** `{rec['default_provider']}`  ")
    lines.append(f"{rec['rationale']}\n")
    lines.append("**Per-operation winners:**\n")
    lines.append("| Operation | Recommended |")
    lines.append("| :-- | :-- |")
    for op, who in rec["per_operation"].items():
        lines.append(f"| `{op}` | {who or '—'} |")
    lines.append("\n> `extract_text` uses field accuracy as a proxy until reference "
                 "transcripts are added to the dataset.\n")

    lines.append("## Composite Ranking\n")
    lines.append("| Rank | Candidate | Composite | Field Acc | Req. Recall | "
                 "Classify | Schema | Portrait | p50 ms | GPU MB | Cost | License |")
    lines.append("| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |")
    for i, c in enumerate(cands, 1):
        lines.append(
            f"| {i} | `{c['candidate']}` | **{_fmt(c['composite_score'])}** | "
            f"{_fmt(c['field_accuracy'])} | {_fmt(c['required_recall'])} | "
            f"{_fmt(c['classification_accuracy'])} | {_fmt(c['schema_following_rate'])} | "
            f"{_fmt(c['portrait_success_rate'])} | {_fmt(c['latency_p50_ms'])} | "
            f"{_fmt(c['gpu_memory_mb'])} | {c['cost_estimate']} | {c['license_status']} |"
        )
    lines.append("")

    lines.append("## Field Accuracy by Document Type\n")
    doc_types = sorted({dt for c in cands for dt in c["by_document_type"].keys()})
    header = "| Candidate | " + " | ".join(doc_types) + " |"
    lines.append(header)
    lines.append("| :-- " + "| :-- " * len(doc_types) + "|")
    for c in cands:
        row = [f"`{c['candidate']}`"] + [
            _fmt(c["by_document_type"].get(dt)) for dt in doc_types
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Robustness — Field Accuracy by Capture Condition\n")
    conds = sorted({cc for c in cands for cc in c["by_capture_condition"].keys()})
    lines.append("| Candidate | " + " | ".join(conds) + " |")
    lines.append("| :-- " + "| :-- " * len(conds) + "|")
    for c in cands:
        row = [f"`{c['candidate']}`"] + [
            _fmt(c["by_capture_condition"].get(cc)) for cc in conds
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Scoring Weights\n")
    lines.append("```json")
    lines.append(json.dumps(summary["weights"], indent=2))
    lines.append("```")
    lines.append(f"\nLatency sub-score reaches 0 at {summary['latency_budget_ms']} ms (p50). "
                 "All weights are tunable in `report.py`.\n")
    return "\n".join(lines)


def write_report(
    results: list[BenchmarkResult], out_dir: str | Path, meta: Optional[dict] = None
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(results)
    json_path = out_dir / "summary.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(render_markdown(summary, meta))
    return json_path, md_path
