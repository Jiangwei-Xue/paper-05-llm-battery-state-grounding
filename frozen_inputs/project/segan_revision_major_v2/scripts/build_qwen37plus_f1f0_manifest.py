#!/usr/bin/env python3
"""Build an offline SHA-256 manifest for a completed Qwen3.7-Plus run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    root = Path(__file__).resolve().parents[1]
    project = root.parent
    files = sorted(path for path in args.run_dir.rglob("*") if path.is_file() and path.name != "QWEN37PLUS_HASH_MANIFEST.jsonl")
    fixed = [
        root / "protocol/qwen37plus_f1f0_v1/QWEN37PLUS_F1F0_PROTOCOL_V1.yaml",
        root / "protocol/qwen37plus_f1f0_v1/QWEN37PLUS_F1F0_PROTOCOL_V1.md",
        root / "protocol/E2B_PROMPT_TEMPLATE_V2.txt",
        root / "protocol/E2B_SCHEMA_V2.json",
        root / "scripts/e2b_parser_v2.py",
        root / "scripts/e2b_v2_common.py",
        root / "scripts/run_qwen37plus_f1f0_v1.py",
        root / "scripts/analyze_qwen37plus_f1f0_v1.py",
        root / "scripts/verify_qwen37plus_f1f0_v1.py",
        root / "scripts/build_qwen37plus_f1f0_manifest.py",
        root / "reviews/qwen37plus_f1f0_v1/QWEN37PLUS_PRE_CALL_MANIFEST.json",
        root / "reviews/qwen37plus_f1f0_v1/QWEN37PLUS_FORMAL_RUN_PLAN.jsonl",
        root / "reviews/qwen37plus_f1f0_v1/QWEN37PLUS_SMOKE_RUN_PLAN.jsonl",
    ]
    entries: list[dict[str, Any]] = []
    for path in [*fixed, *files]:
        if not path.is_file():
            raise SystemExit(f"missing manifest path: {path}")
        entries.append({"path": str(path.relative_to(project)), "sha256": sha256(path), "size_bytes": path.stat().st_size, "kind": "run_evidence" if path in files else "frozen_control"})
    out = args.run_dir / "QWEN37PLUS_HASH_MANIFEST.jsonl"
    out.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for item in entries), encoding="utf-8")
    print(json.dumps({"status": "PASS", "entries": len(entries), "run_evidence_files": len(files), "manifest": str(out.relative_to(project))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
