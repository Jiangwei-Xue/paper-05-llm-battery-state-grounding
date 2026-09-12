#!/usr/bin/env python3
"""Compute E2 suffix-projection sidecars from saved provider outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.hashing import sha256_json  # noqa: E402

from energy_agent_reliability.config import load_protocol  # noqa: E402
from energy_agent_reliability.online_gate_v6 import lexicographic_projection  # noqa: E402

BRANCHES = ("C", "M", "S", "K", "C2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--one-record")
    parser.add_argument("--one-branch")
    parser.add_argument("--one-output")
    args = parser.parse_args()
    if args.one_record:
        if not args.one_branch or not args.one_output:
            raise SystemExit("single-item mode requires branch and output")
        record = json.loads(Path(args.one_record).read_text())
        _write(Path(args.one_output), _project(record, args.one_branch))
        return
    if not 1 <= args.workers <= 4:
        raise SystemExit("E2 projection workers must be between one and four")
    run_dir = ROOT / "runs/experiments_v2/e2" / args.run_label
    records = [
        json.loads(path.read_text()) for path in sorted((run_dir / "records").glob("*.json"))
    ]
    if len(records) != 360:
        raise SystemExit(f"E2 projection requires 360 saved blocks, found {len(records)}")
    sidecar_dir = run_dir / "projection_sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    work = [
        (record, branch, run_dir / "records" / f"{record['block_id']}.json")
        for record in records
        for branch in BRANCHES
        if not (sidecar_dir / f"{record['block_id']}__{branch}.json").exists()
    ]
    completed = 1800 - len(work)
    queue = list(work)
    active: list[dict[str, Any]] = []
    while queue or active:
        while queue and len(active) < args.workers:
            record, branch, record_path = queue.pop(0)
            output = sidecar_dir / f"{record['block_id']}__{branch}.json"
            if not record["branches"][branch]["parsed"]["ok"]:
                _write(output, _failure(record, branch, "not_applicable_parser_failure"))
                completed += 1
                print(f"e2_projection={completed}/1800", flush=True)
                continue
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-label",
                args.run_label,
                "--one-record",
                str(record_path),
                "--one-branch",
                branch,
                "--one-output",
                str(output),
            ]
            active.append(
                {
                    "process": subprocess.Popen(command, cwd=ROOT),
                    "started": time.monotonic(),
                    "started_utc": datetime.now(UTC).isoformat(),
                    "record": record,
                    "branch": branch,
                    "output": output,
                }
            )
        time.sleep(0.1)
        for item in list(active):
            process: subprocess.Popen[bytes] = item["process"]
            elapsed = time.monotonic() - float(item["started"])
            if process.poll() is None and elapsed < 120.0:
                continue
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                _write(
                    item["output"],
                    _failure(
                        item["record"],
                        item["branch"],
                        "hard_timeout_120s",
                        started_utc=item["started_utc"],
                        finished_utc=datetime.now(UTC).isoformat(),
                        timeout_seconds=elapsed,
                        termination_signal="SIGTERM",
                    ),
                )
            elif process.returncode != 0 or not item["output"].exists():
                _write(
                    item["output"],
                    _failure(item["record"], item["branch"], f"worker_exit_{process.returncode}"),
                )
            active.remove(item)
            completed += 1
            print(f"e2_projection={completed}/1800", flush=True)
    summary = {
        "schema_version": "experiments_v2_e2_projection_summary_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "blocks": len(records),
        "sidecars": len(list(sidecar_dir.glob("*.json"))),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    _write(run_dir / "E2_PROJECTION_SUMMARY.json", summary)


def _project(record: dict[str, Any], branch: str) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        parsed = record["branches"][branch]["parsed"]
        if not parsed["ok"]:
            return _failure(record, branch, "not_applicable_parser_failure")
        battery = load_protocol(ROOT / "configs/frozen_protocol.yaml").battery.model_copy(
            update={"terminal_soc_kwh": 325.0}
        )
        task = record["task"]
        activation = int(task["visible_update"]["activation_step"])
        solution = lexicographic_projection(
            task,
            [float(value) for value in parsed["dense_action_kw"]],
            battery,
            initial_soc_kwh=float(task["v2_prefix_intervention"]["canonical_event_soc_kwh"]),
            start_step=activation,
            apply_event=True,
            require_terminal=True,
        )
        payload = {
            "schema_version": "experiments_v2_e2_projection_sidecar_v1",
            "block_id": record["block_id"],
            "branch": branch,
            "record_evidence_root_sha256": record["evidence_root_sha256"],
            "status": "computed",
            "projection_distance_kwh": solution.distance_kwh,
            "throughput_kwh": solution.throughput_kwh,
            "physical_cost_usd": solution.physical_cost_usd,
            "solver_status": solution.solver_status,
            "simultaneous_intervals": solution.simultaneous_intervals,
            "maximum_simultaneous_kw": solution.maximum_simultaneous_kw,
            "projected_action_kw": solution.action_kw.tolist(),
            "provider_calls_performed": 0,
            "network_attempts": 0,
        }
        payload["sidecar_sha256"] = sha256_json(payload)
        return payload
    except Exception as exc:
        return _failure(record, branch, type(exc).__name__)


def _failure(
    record: dict[str, Any],
    branch: str,
    reason: str,
    *,
    started_utc: str | None = None,
    finished_utc: str | None = None,
    timeout_seconds: float | None = None,
    termination_signal: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "experiments_v2_e2_projection_sidecar_v1",
        "block_id": record["block_id"],
        "branch": branch,
        "record_evidence_root_sha256": record["evidence_root_sha256"],
        "status": "not_computed",
        "reason": reason,
        "failure_input_sha256": sha256_json(
            {
                "record_evidence_root_sha256": record["evidence_root_sha256"],
                "branch": branch,
                "parsed_action_sha256": sha256_json(record["branches"][branch].get("parsed", {})),
                "task_sha256": sha256_json(record["task"]),
            }
        ),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "timeout_seconds": timeout_seconds,
        "termination_signal": termination_signal,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    payload["sidecar_sha256"] = sha256_json(payload)
    return payload


def _write(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    partial.replace(path)


if __name__ == "__main__":
    main()
