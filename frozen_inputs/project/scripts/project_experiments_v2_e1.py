#!/usr/bin/env python3
"""Compute E1 projection sidecars from saved provider outputs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments_v2/src"))

from experiments_v2.hashing import sha256_json  # noqa: E402

from energy_agent_reliability.config import load_protocol  # noqa: E402
from energy_agent_reliability.online_gate_v6 import (  # noqa: E402
    lexicographic_projection,
    stage1_information_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("E1 projection workers must be between one and four")
    run_dir = ROOT / "runs/experiments_v2/e1" / args.run_label
    records = [json.loads(path.read_text()) for path in sorted((run_dir / "records").glob("*.json"))]
    if len(records) != 240:
        raise SystemExit(f"E1 projection requires 240 saved records, found {len(records)}")
    sidecar_dir = run_dir / "projection_sidecars"
    sidecar_dir.mkdir(exist_ok=True)
    pending = [
        record
        for record in records
        if not (sidecar_dir / f"{record['episode_id']}.json").exists()
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_project_record, record): record for record in pending}
        completed = len(records) - len(pending)
        for future in as_completed(futures):
            record = futures[future]
            try:
                sidecar = future.result()
            except Exception as exc:
                sidecar = _failure_sidecar(record, f"worker:{type(exc).__name__}")
            _write(sidecar_dir / f"{record['episode_id']}.json", sidecar)
            completed += 1
            print(f"e1_projection={completed}/240", flush=True)
    summary = {
        "schema_version": "experiments_v2_e1_projection_summary_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "records": len(records),
        "sidecars": len(list(sidecar_dir.glob("*.json"))),
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    _write(run_dir / "E1_PROJECTION_SUMMARY.json", summary)


def _project_record(record: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(120)
    try:
        if not record["outcome"]["parser_success"]:
            return _failure_sidecar(record, "not_applicable_parser_failure")
        protocol = load_protocol(ROOT / "configs/frozen_protocol.yaml")
        battery = protocol.battery.model_copy(update={"terminal_soc_kwh": 325.0})
        task = record["task"]
        activation = int(record["canonical_context"]["activation"])
        stage1 = [float(value) for value in record["stage1"]["parsed"]["dense_action_kw"]]
        stage2 = [float(value) for value in record["stage2"]["parsed"]["dense_action_kw"]]
        first = lexicographic_projection(
            task,
            stage1,
            battery,
            initial_soc_kwh=battery.initial_soc_kwh,
            start_step=0,
            apply_event=False,
            require_terminal=True,
            frame_override=stage1_information_frame(task),
        )
        second = lexicographic_projection(
            task,
            stage2,
            battery,
            initial_soc_kwh=float(record["canonical_context"]["event_soc_kwh"]),
            start_step=activation,
            apply_event=True,
            require_terminal=True,
        )
        payload = {
            "schema_version": "experiments_v2_e1_projection_sidecar_v1",
            "episode_id": record["episode_id"],
            "record_evidence_root_sha256": record["evidence_root_sha256"],
            "status": "computed",
            "projection_distance_kwh": first.distance_kwh + second.distance_kwh,
            "stage1_projection": _solution(first),
            "stage2_projection": _solution(second),
            "provider_calls_performed": 0,
            "network_attempts": 0,
        }
        payload["sidecar_sha256"] = sha256_json(payload)
        return payload
    except Exception as exc:
        return _failure_sidecar(record, type(exc).__name__)
    finally:
        signal.alarm(0)


def _solution(value: Any) -> dict[str, Any]:
    return {
        "action_kw": value.action_kw.tolist(),
        "distance_kwh": value.distance_kwh,
        "throughput_kwh": value.throughput_kwh,
        "physical_cost_usd": value.physical_cost_usd,
        "solver_status": value.solver_status,
        "simultaneous_intervals": value.simultaneous_intervals,
        "maximum_simultaneous_kw": value.maximum_simultaneous_kw,
    }


def _failure_sidecar(record: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = {
        "schema_version": "experiments_v2_e1_projection_sidecar_v1",
        "episode_id": record["episode_id"],
        "record_evidence_root_sha256": record["evidence_root_sha256"],
        "status": "not_computed",
        "reason": reason,
        "provider_calls_performed": 0,
        "network_attempts": 0,
    }
    payload["sidecar_sha256"] = sha256_json(payload)
    return payload


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("projection exceeded 120 seconds")


def _write(path: Path, value: Any) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    main()
