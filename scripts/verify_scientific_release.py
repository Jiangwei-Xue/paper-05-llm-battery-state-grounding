#!/usr/bin/env python3
"""Verify the scientific contract of the extracted reproduction package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    required = [
        "README.md", "RELEASE_AUDIT.md", "EXPERIMENT_TIMELINE.md",
        "METHOD_VERSION_HISTORY.md", "experiments_manifest.json",
        "results_manifest.json", "manifests/PUBLIC_RECORD_MANIFEST.jsonl",
        "docs/experiment_design.md", "docs/reproduction_protocol.md",
        "docs/paper_results_map.md", "protocol_code/experiments_v2/p0.py",
        "protocol_code/e2b_parser_v2.py", "tests/test_parsers.py",
        "tests/test_release_contract.py", "paper_reference/main.tex",
        "paper_reference/state_grounding_preprint_v5_ccby_github_20260912.pdf",
        "experiment_scaffolding/EXPERIMENT_MAP.md",
        "experiment_scaffolding/EXPERIMENT_MAP.json",
        "experiment_scaffolding/LIVE_RUNNER_GUIDE.md",
        "experiment_scaffolding/RUNNER_INDEX.json",
        "runtime/encoded/RUNTIME_MATERIALIZATION_MANIFEST.json",
        "scripts/materialize_runtime.py",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            errors.append(f"missing:{relative}")

    paper = load(ROOT / "LATEST_PAPER.json")
    for item in paper["files"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            errors.append(f"paper_hash:{item['path']}")

    experiments = load(ROOT / "experiments_manifest.json")
    for experiment in experiments["experiments"]:
        for key in ("records_path", "run_plan", "task_manifest", "admission_manifest"):
            value = experiment.get(key)
            if value and not (ROOT / value).exists():
                errors.append(f"experiment_path:{value}")

    results = load(ROOT / "results_manifest.json")
    for table in results["core_tables"]:
        for value in table["sources"]:
            if not (ROOT / value).is_file():
                errors.append(f"result_source:{value}")

    runner_index = load(ROOT / "experiment_scaffolding/RUNNER_INDEX.json")
    expected_experiments = {"E1", "E2", "F1", "F0", "QWEN37"}
    observed_experiments = {row["experiment"] for row in runner_index["experiments"]}
    if observed_experiments != expected_experiments:
        errors.append("runner_index_experiment_population")
    for experiment in runner_index["experiments"]:
        for key in ("protocol", "pre_call_manifest", "runner"):
            relative = experiment.get(key)
            if not relative or not (ROOT / relative).is_file():
                errors.append(f"runner_index_path:{experiment.get('experiment')}:{key}:{relative}")

    runtime_manifest = load(ROOT / "runtime/encoded/RUNTIME_MATERIALIZATION_MANIFEST.json")
    if not runtime_manifest.get("files"):
        errors.append("runtime_materialization_manifest_empty")
    for item in runtime_manifest.get("files", []):
        encoded = ROOT / "runtime/encoded" / item["encoded_path"]
        if not encoded.is_file() or sha256(encoded) != item["encoded_sha256"]:
            errors.append(f"runtime_materialization_hash:{item['encoded_path']}")

    record_rows = [
        json.loads(line)
        for line in (ROOT / "manifests/PUBLIC_RECORD_MANIFEST.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(record_rows) != 825 or len({row["path"] for row in record_rows}) != 825:
        errors.append("record_manifest_population")
    for row in record_rows:
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"record_hash:{row['path']}")

    report = {
        "schema_version": "state_grounding_scientific_release_v1",
        "status": "PASS" if not errors else "HOLD",
        "paper_files_hash_verified": len(paper["files"]),
        "experiments_verified": len(experiments["experiments"]),
        "paper_tables_mapped": len(results["core_tables"]),
        "record_files_hash_verified": len(record_rows),
        "experiment_runner_families_verified": len(observed_experiments),
        "encoded_runtime_files_verified": len(runtime_manifest.get("files", [])),
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
