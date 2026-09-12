"""Command-line entry points used by the provider-free runbook."""

from __future__ import annotations

import argparse
import json

from .freeze import freeze_manifest
from .runner import dry_run


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--output", default="experiments_v2/reports/dry_run_artifacts")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", default="experiments_v2")
    freeze.add_argument("--output", default="experiments_v2/manifests/HASH_MANIFEST.json")
    args = parser.parse_args()
    if args.command == "dry-run":
        print(json.dumps(dry_run(args.output), indent=2, sort_keys=True))
    else:
        print(json.dumps(freeze_manifest(args.root, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
