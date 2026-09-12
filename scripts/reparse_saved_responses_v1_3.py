#!/usr/bin/env python3
"""Re-run the frozen parsers against every retained provider response.

This verifier performs no provider or network call.  It derives each parsing
contract from the retained task and interface, invokes the exact archived
parser source shipped in ``protocol_code/``, and compares the complete parsed
object with the value saved in the experimental record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "frozen_inputs/project"
PROTOCOL_CODE = ROOT / "protocol_code"
sys.path.insert(0, str(PROTOCOL_CODE))

from e2b_parser_v2 import parse_output as parse_e2b_v2  # noqa: E402
from experiments_v2.p0 import parse_output as parse_p0  # noqa: E402
from experiments_v2.p0 import parsed_payload  # noqa: E402

DATASETS = {
    "E1": WORKSPACE / "runs/experiments_v2/e1/e1_20260801T044500Z/records",
    "E2": WORKSPACE / "runs/experiments_v2/e2/e2_20260801T054708Z/records",
    "F1": WORKSPACE
    / "segan_revision_major_v2/runs/e2b_protocol_v2/e2b_v2_f1_20260813/records",
    "F0": WORKSPACE
    / "segan_revision_major_v2/runs/f0_formal_v1/f0_formal_20260820T_authorized_v1/records",
    "QWEN37": WORKSPACE
    / "segan_revision_major_v2/runs/qwen37plus_f1f0_v1/qwen37plus_formal_20260826T/records",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(
    *,
    dataset: str,
    record_id: str,
    call_id: str,
    observed: dict[str, Any],
    saved: Any,
    errors: list[dict[str, Any]],
) -> None:
    if canonical(observed) != canonical(saved):
        errors.append(
            {
                "dataset": dataset,
                "record_id": record_id,
                "call_id": call_id,
                "observed_sha256": hashlib.sha256(canonical(observed)).hexdigest(),
                "saved_sha256": hashlib.sha256(canonical(saved)).hexdigest(),
                "reason": "parsed_object_mismatch",
            }
        )


def p0_horizon(task: dict[str, Any], stage_name: str) -> int:
    episode = task["model_visible_episode"]
    if stage_name == "stage1":
        return len(episode["stage_1"]["visible_timeseries"])
    return len(episode["stage_2"]["remaining_timeseries"])


def verify_e1_or_e2(
    dataset: str,
    directory: Path,
    errors: list[dict[str, Any]],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    counts: Counter[str] = Counter()
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = load(path)
        record_id = str(record.get("episode_id") or record.get("block_id") or path.stem)
        task = record["task"]
        tasks[str(record["task_sha256"])] = task
        interface = str(record["interface"])
        calls: list[tuple[str, dict[str, Any]]] = [("stage1", record["stage1"])]
        if dataset == "E1":
            calls.append(("stage2", record["stage2"]))
        else:
            calls.extend((name, item) for name, item in record["branches"].items())
        for call_id, item in calls:
            counts["retained_calls"] += 1
            response = item.get("response", {})
            if response.get("status") != "ok":
                counts["not_applicable_provider_failure_or_skip"] += 1
                continue
            raw_text = response.get("content")
            if not isinstance(raw_text, str):
                errors.append(
                    {
                        "dataset": dataset,
                        "record_id": record_id,
                        "call_id": call_id,
                        "reason": "successful_response_content_missing",
                    }
                )
                continue
            stage_kind = "stage1" if call_id == "stage1" else "stage2"
            observed = parsed_payload(
                parse_p0(raw_text, interface, p0_horizon(task, stage_kind))
            )
            compare(
                dataset=dataset,
                record_id=record_id,
                call_id=call_id,
                observed=observed,
                saved=item.get("parsed"),
                errors=errors,
            )
            counts["successful_responses_reparsed"] += 1
    return dict(counts), tasks


def verify_four_branch(
    dataset: str,
    directory: Path,
    tasks_by_hash: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for path in sorted(directory.glob("*.json")):
        record = load(path)
        record_id = str(record.get("block_id") or path.stem)
        task_hash = str(record.get("task_sha256"))
        task = tasks_by_hash.get(task_hash)
        if task is None:
            errors.append(
                {
                    "dataset": dataset,
                    "record_id": record_id,
                    "reason": "task_hash_join_failed",
                    "task_sha256": task_hash,
                }
            )
            continue
        horizon = p0_horizon(task, "stage2")
        for call_id, item in record["branches"].items():
            counts["retained_calls"] += 1
            response = item.get("response", {})
            if response.get("status") != "ok":
                counts["not_applicable_provider_failure_or_skip"] += 1
                continue
            raw_text = response.get("content")
            if not isinstance(raw_text, str):
                errors.append(
                    {
                        "dataset": dataset,
                        "record_id": record_id,
                        "call_id": call_id,
                        "reason": "successful_response_content_missing",
                    }
                )
                continue
            parsed = parse_e2b_v2(raw_text, horizon)
            observed = {
                "ok": parsed.ok,
                "raw_actions": parsed.raw_actions,
                "dense_action_kw": parsed.dense_action_kw,
                "diagnostics": parsed.diagnostics,
                "horizon": parsed.horizon,
                "segment_count": parsed.segment_count,
            }
            compare(
                dataset=dataset,
                record_id=record_id,
                call_id=call_id,
                observed=observed,
                saved=item.get("parsed"),
                errors=errors,
            )
            counts["successful_responses_reparsed"] += 1
    return dict(counts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[dict[str, Any]] = []

    e1, _ = verify_e1_or_e2("E1", DATASETS["E1"], errors)
    e2, tasks_by_hash = verify_e1_or_e2("E2", DATASETS["E2"], errors)
    populations = {
        "E1": e1,
        "E2": e2,
        "F1": verify_four_branch("F1", DATASETS["F1"], tasks_by_hash, errors),
        "F0": verify_four_branch("F0", DATASETS["F0"], tasks_by_hash, errors),
        "QWEN37": verify_four_branch(
            "QWEN37", DATASETS["QWEN37"], tasks_by_hash, errors
        ),
    }
    reparsed = sum(item.get("successful_responses_reparsed", 0) for item in populations.values())
    not_applicable = sum(
        item.get("not_applicable_provider_failure_or_skip", 0)
        for item in populations.values()
    )
    retained = sum(item.get("retained_calls", 0) for item in populations.values())
    parser_hashes = {
        "protocol_code/experiments_v2/p0.py": digest_file(
            PROTOCOL_CODE / "experiments_v2/p0.py"
        ),
        "protocol_code/experiments_v2/actions.py": digest_file(
            PROTOCOL_CODE / "experiments_v2/actions.py"
        ),
        "protocol_code/experiments_v2/hashing.py": digest_file(
            PROTOCOL_CODE / "experiments_v2/hashing.py"
        ),
        "protocol_code/e2b_parser_v2.py": digest_file(
            PROTOCOL_CODE / "e2b_parser_v2.py"
        ),
    }
    status = "PASS" if not errors else "HOLD"
    report = {
        "schema_version": "state_grounding_parser_replay_v1_3",
        "status": status,
        "retained_calls": retained,
        "successful_responses_reparsed": reparsed,
        "not_applicable_provider_failure_or_skip": not_applicable,
        "parsed_object_mismatches": len(errors),
        "populations": populations,
        "parser_source_sha256": parser_hashes,
        "provider_calls_performed": 0,
        "network_attempts": 0,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "retained_calls": retained,
                "reparsed": reparsed,
                "not_applicable": not_applicable,
                "mismatches": len(errors),
                "provider_calls": 0,
                "network_attempts": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
