"""Candidate manifest construction with stable IDs and separated scorer oracles."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FrozenProtocol
from .episode import build_episode_surface
from .events import generate_event
from .gates import require_formal_freeze_gate
from .provenance import write_jsonl
from .state_methods import method_surface_definitions


def stable_scenario_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"scn_{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"


def encoded_scenario_id(identity: dict[str, Any]) -> str:
    """Create a stable ID that exposes the pre-registered design cell."""
    digest = stable_scenario_id(identity).removeprefix("scn_")[:10]
    date_token = str(identity["start_timestamp"])[:10].replace("-", "")
    version_token = _slug(str(identity["protocol_version"]))
    return "_".join(
        [
            "scn",
            version_token,
            _slug(str(identity["location_id"])),
            _slug(str(identity["season"])),
            date_token,
            _slug(str(identity["event_family"])),
            _slug(str(identity["difficulty"])),
            f"r{int(identity['replicate'])}",
            digest,
        ]
    )


def build_candidate_manifest(
    protocol: FrozenProtocol,
    processed_dir: str | Path,
    manifest_dir: str | Path,
    oracle_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Build the public candidate manifest and write hidden oracles separately."""
    processed = Path(processed_dir)
    require_formal_freeze_gate(processed.parent.parent)
    manifest_path = Path(manifest_dir)
    oracle_path = Path(oracle_dir) if oracle_dir is not None else manifest_path / "oracles"
    oracle_path.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for location in protocol.locations:
        source = processed / f"{location.location_id}_{protocol.frozen_year}_15min.parquet"
        if not source.exists():
            raise FileNotFoundError(f"Missing processed source: {source}")
        site_frame = _load_site_frame(source)
        for season, months in protocol.scenario.seasons.items():
            for family in protocol.scenario.event_families:
                for difficulty in protocol.scenario.difficulties:
                    for replicate in range(protocol.scenario.candidates_per_cell):
                        start = _choose_start(protocol, season, months, location.location_id, family, difficulty, replicate)
                        event_seed = _stable_seed(
                            protocol.selection.event_generation_seed,
                            location.location_id,
                            season,
                            family,
                            difficulty,
                            replicate,
                        )
                        event = generate_event(family, difficulty, event_seed, protocol.battery)
                        identity = {
                            "protocol_version": protocol.protocol_version,
                            "location_id": location.location_id,
                            "season": season,
                            "event_family": family,
                            "difficulty": difficulty,
                            "start_timestamp": start.isoformat(),
                            "replicate": replicate,
                            "visible_update": event.visible_update,
                        }
                        scenario_id = encoded_scenario_id(identity)
                        candidate = {
                            "scenario_id": scenario_id,
                            **identity,
                            "horizon_hours": protocol.scenario.horizon_hours,
                            "source_data": str(source),
                            "oracle_ref": str((oracle_path / f"{scenario_id}.json").relative_to(oracle_path.parent)),
                            "state_methods": method_surface_definitions(),
                        }
                        frame = _slice_frame(site_frame, start, int(protocol.scenario.horizon_hours))
                        candidate["model_visible_episode"] = build_episode_surface(protocol, candidate, frame)
                        oracle = {
                            "scenario_id": scenario_id,
                            "scorer_oracle": event.scorer_oracle,
                            "created_for_protocol": protocol.protocol_version,
                        }
                        (oracle_path / f"{scenario_id}.json").write_text(
                            json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                        )
                        candidates.append(candidate)
    candidates.sort(key=lambda item: item["scenario_id"])
    if len({candidate["scenario_id"] for candidate in candidates}) != len(candidates):
        raise ValueError("Stable-ID collision in candidate manifest.")
    write_jsonl(manifest_path / "candidates.jsonl", candidates)
    return candidates


def scenario_frame(candidate: dict[str, Any]) -> pd.DataFrame:
    full = _load_site_frame(Path(candidate["source_data"]))
    start = pd.Timestamp(candidate["start_timestamp"])
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    return _slice_frame(full, start, int(candidate["horizon_hours"]), str(candidate["scenario_id"]))


def _choose_start(
    protocol: FrozenProtocol,
    season: str,
    months: list[int],
    location_id: str,
    family: str,
    difficulty: str,
    replicate: int,
) -> pd.Timestamp:
    candidate_days: list[date] = []
    for month in months:
        for day in range(3, 22):
            current = date(protocol.frozen_year, month, day)
            candidate_days.append(current)
    seed = _stable_seed(protocol.selection.event_generation_seed, location_id, season, family, difficulty, replicate)
    chosen = random.Random(seed).choice(candidate_days)
    return pd.Timestamp(chosen, tz="UTC")


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _load_site_frame(source: Path) -> pd.DataFrame:
    full = pd.read_parquet(source)
    if "timestamp_utc" in full.columns:
        full.index = pd.DatetimeIndex(pd.to_datetime(full["timestamp_utc"], utc=True))
    else:
        full.index = pd.DatetimeIndex(pd.to_datetime(full.index, utc=True))
    return full


def _slice_frame(
    full: pd.DataFrame, start: pd.Timestamp, horizon_hours: int, scenario_id: str = "candidate"
) -> pd.DataFrame:
    stop = start + timedelta(hours=horizon_hours)
    frame = full.loc[(full.index >= start) & (full.index < stop)].copy()
    expected_steps = horizon_hours * 4
    if len(frame) != expected_steps:
        raise ValueError(f"{scenario_id} has {len(frame)} rather than requested horizon rows.")
    return frame


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")[:48]
