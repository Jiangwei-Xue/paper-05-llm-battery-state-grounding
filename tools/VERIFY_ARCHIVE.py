#!/usr/bin/env python3
"""Command-line entry point for the canonical experiment release workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from release_core import (
    TOOL_VERSION,
    ReleaseError,
    build_release,
    init_project,
    json_text,
    sanitize_runtime_text,
    verify_archive,
)


def print_result(result: dict[str, object]) -> None:
    print(json_text(result), end="")
    for name in (
        "INTEGRITY_GATE",
        "PRIVACY_GATE",
        "DOCUMENTATION_GATE",
        "ARCHIVE_HYGIENE_GATE",
        "REPRODUCIBILITY_GATE",
        "OVERALL_RELEASE_STATUS",
    ):
        value = result.get(name)
        if isinstance(value, str):
            print(f"{name}={value}")


def print_public_result(result: dict[str, object]) -> None:
    """Print a neutral public projection while preserving fail-closed status."""
    overall = result.get("OVERALL_RELEASE_STATUS")
    checks = {
        "Package integrity": result.get("INTEGRITY_GATE"),
        "Documentation": result.get("DOCUMENTATION_GATE"),
        "Archive integrity": result.get("ARCHIVE_HYGIENE_GATE"),
        "Reproducibility": result.get("REPRODUCIBILITY_GATE"),
    }
    for label, status in checks.items():
        public_status = status if status in {"PASS", "FAIL", "NOT_VERIFIED"} else "NOT_VERIFIED"
        print(f"[{public_status}] {label}")
    print(f"PACKAGE VALIDATION: {'PASS' if overall == 'PASS' else 'FAIL'}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="experiment-release",
        description="Build and verify staging-first, privacy-bounded experiment archives.",
    )
    result.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    subparsers = result.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build, duplicate-build, extract, and verify configured archives")
    build.add_argument("--config", required=True, type=Path)

    audit = subparsers.add_parser("audit-only", aliases=["dry-run"], help="Create an isolated staging tree and run preflight gates only")
    audit.add_argument("--config", required=True, type=Path)

    verify = subparsers.add_parser("verify", help="Verify a final archive and its clean extraction")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--public", action="store_true", help="Print a neutral public validation summary")

    initialize = subparsers.add_parser("init", help="Add thin canonical-tool integration files to a project")
    initialize.add_argument("--project-root", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "build":
            value = build_release(args.config, audit_only=False)
        elif args.command in {"audit-only", "dry-run"}:
            value = build_release(args.config, audit_only=True)
        elif args.command == "verify":
            value = verify_archive(args.archive)
        elif args.command == "init":
            template_dir = Path(__file__).resolve().parent / "templates"
            value = init_project(args.project_root, template_dir)
        else:
            raise ReleaseError("unsupported command")
    except ReleaseError as exc:
        value = {
            "tool_version": TOOL_VERSION,
            "error": sanitize_runtime_text(str(exc)),
            "INTEGRITY_GATE": "NOT_VERIFIED",
            "PRIVACY_GATE": "NOT_VERIFIED",
            "DOCUMENTATION_GATE": "NOT_VERIFIED",
            "ARCHIVE_HYGIENE_GATE": "NOT_VERIFIED",
            "REPRODUCIBILITY_GATE": "NOT_VERIFIED",
            "OVERALL_RELEASE_STATUS": "FAIL",
        }
    except Exception as exc:  # Fail closed without printing a machine-local traceback.
        value = {
            "tool_version": TOOL_VERSION,
            "error": f"unexpected failure: {sanitize_runtime_text(str(exc))}",
            "INTEGRITY_GATE": "NOT_VERIFIED",
            "PRIVACY_GATE": "NOT_VERIFIED",
            "DOCUMENTATION_GATE": "NOT_VERIFIED",
            "ARCHIVE_HYGIENE_GATE": "NOT_VERIFIED",
            "REPRODUCIBILITY_GATE": "NOT_VERIFIED",
            "OVERALL_RELEASE_STATUS": "FAIL",
        }
    if args.command == "verify" and args.public:
        print_public_result(value)
    else:
        print_result(value)
    return 0 if value.get("OVERALL_RELEASE_STATUS") == "PASS" or value.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
