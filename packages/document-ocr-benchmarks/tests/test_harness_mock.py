"""End-to-end pipeline test using the synthetic generator + mock provider.

Exercises generate -> load -> analyze -> score -> report without any OCR backend.
"""

import json
from pathlib import Path

from document_ocr_benchmarks.harness import run_benchmark
from document_ocr_benchmarks.report import build_summary, write_report
from document_ocr_benchmarks.models import CaptureCondition, DocumentType
from document_ocr_benchmarks.synth.generator import generate_dataset


def test_pipeline_end_to_end(tmp_path: Path):
    dataset = tmp_path / "ds"
    generate_dataset(
        dataset,
        doc_types=[DocumentType.NIN_SLIP, DocumentType.UTILITY_BILL],
        conditions=[CaptureCondition.CLEAN],
        per_type=1,
    )
    manifest = dataset / "manifests" / "synthetic_v1.json"
    assert manifest.exists()

    run = run_benchmark(
        manifest_path=manifest,
        dataset_root=dataset,
        providers=["mock"],
        results_root=tmp_path / "results",
        progress=False,
    )
    assert len(run.results) == 2
    for r in run.results:
        assert r.candidate == "mock"
        assert 0.0 <= r.field_accuracy <= 1.0
        assert r.latency_ms >= 0.0

    summary = build_summary(run.results)
    assert summary["recommendation"]["default_provider"] == "mock"

    json_path, md_path = write_report(run.results, tmp_path / "results" / run.run_id, run.meta)
    assert json_path.exists() and md_path.exists()
    assert "Recommendation" in md_path.read_text()


def test_results_jsonl_is_camel_case(tmp_path: Path):
    dataset = tmp_path / "ds"
    generate_dataset(dataset, doc_types=[DocumentType.NIN_SLIP],
                     conditions=[CaptureCondition.CLEAN], per_type=1)
    run = run_benchmark(
        manifest_path=dataset / "manifests" / "synthetic_v1.json",
        dataset_root=dataset, providers=["mock"],
        results_root=tmp_path / "results", progress=False,
    )
    row = run.results[0].model_dump(by_alias=True)
    assert "fieldAccuracy" in row
    assert "requiredFieldRecall" in row
