#!/usr/bin/env python3
"""Materialize the byte-exact offline runtime into a disposable output tree."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--encoded", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(args.base, args.output)
    manifest_path = args.encoded / "RUNTIME_MATERIALIZATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restored = 0
    for item in manifest["files"]:
        payload = args.encoded / item["encoded_path"]
        if sha256(payload) != item["encoded_sha256"]:
            raise SystemExit(f"encoded runtime hash mismatch: {item['encoded_path']}")
        target = args.output / item["runtime_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(payload.read_bytes(), validate=True))
        if sha256(target) != item["runtime_sha256"]:
            raise SystemExit(f"materialized runtime hash mismatch: {item['runtime_path']}")
        restored += 1
    print(json.dumps({"status": "PASS", "materialized_files": restored}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
