"""Real-data deterministic smoke checks; never a candidate or freeze operation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pvlib  # type: ignore[import-untyped]
from matplotlib import pyplot as plt

from .baselines import perfect_information_baseline, rolling_horizon_baseline
from .battery import simulate_dispatch
from .config import FrozenProtocol, Location
from .contracts import load_system_sizing
from .events import dynamic_override_schedule, generate_event
from .forecasting import revised_pv_forecast
from .provenance import utc_now


def run_deterministic_smoke(
    protocol: FrozenProtocol, project_root: str | Path, site: str
) -> dict[str, Any]:
    """Run fixed normal and stress windows only after raw and processed gates pass."""
    root = Path(project_root)
    readiness = _read_json(root / "reports" / "DATA_READINESS_REPORT.json")
    audit = _read_json(root / "reports" / "PROCESSED_DATA_AUDIT.json")
    if (
        readiness is None
        or readiness.get("status") != "pass"
        or readiness.get("claim_bearing_data_ready") is not True
    ):
        return _blocked_report(protocol, site, "DATA_CONTRACT_V1_2 readiness gate has not passed.")
    if audit is None or audit.get("status") != "pass" or audit.get("passed") is not True:
        return _blocked_report(protocol, site, "Processed-data audit has not passed.")
    source = root / "data" / "processed" / f"{site}_{protocol.frozen_year}_15min.parquet"
    sizing_path = root / "protocol" / "SYSTEM_SIZING_CONTRACT_V1.yaml"
    if not source.exists() or not sizing_path.exists():
        return _blocked_report(protocol, site, "Processed site data or frozen sizing contract is absent.")
    try:
        sizing = load_system_sizing(sizing_path)
        frame = _load_site_table(source)
        location = _location(protocol, site)
        window_reports = [
            _run_window(protocol, kind, window, location, sizing.ac_capacity_kw, root)
            for kind, window in _select_windows(frame)
        ]
        checks = {
            key: all(item["checks"][key] for item in window_reports)
            for key in window_reports[0]["checks"]
        }
        return {
            "report_type": "DETERMINISTIC_SMOKE_REPORT",
            "created_utc": utc_now(),
            "status": "pass" if all(checks.values()) else "fail",
            "site": site,
            "window_selection_rule": "minimum and maximum deterministic PV-ramp-plus-price-variation score over continuous seven-day windows",
            "windows": window_reports,
            "checks": checks,
            "passed": all(checks.values()),
        }
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        return _blocked_report(protocol, site, str(exc))


def write_smoke_report(report: dict[str, Any], project_root: str | Path) -> Path:
    destination = Path(project_root) / "reports" / "DETERMINISTIC_SMOKE_REPORT.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _load_site_table(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {
        "timestamp_utc",
        "load_kw",
        "pv_ac_kw",
        "day_ahead_price_usd_mwh",
        "clear_sky_ghi",
        "forecast_pv_kw",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Prepared data lacks smoke-test columns: {sorted(missing)}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    return frame.sort_index()


def _select_windows(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    length = 7 * 24 * 4
    candidates: list[tuple[float, pd.DataFrame]] = []
    for start in range(0, len(frame) - length + 1, 4):
        window = frame.iloc[start : start + length]
        if not _continuous(window):
            continue
        pv_ramp = float(window["pv_ac_kw"].diff().abs().mean())
        price_variation = float(window["day_ahead_price_usd_mwh"].std())
        score = pv_ramp + price_variation
        candidates.append((score, window))
    if not candidates:
        raise ValueError("No continuous seven-day, 15-minute real-data window is available.")
    normal = min(candidates, key=lambda item: (item[0], item[1].index[0]))[1]
    stress = max(candidates, key=lambda item: (item[0], -item[1].index[0].value))[1]
    return [("normal", normal), ("stress", stress)]


def _continuous(window: pd.DataFrame) -> bool:
    required = ["load_kw", "pv_ac_kw", "day_ahead_price_usd_mwh", "forecast_pv_kw"]
    return bool(
        len(window) == 7 * 24 * 4
        and (window.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
        and np.isfinite(window[required].to_numpy(dtype=float)).all()
    )


def _run_window(
    protocol: FrozenProtocol,
    kind: str,
    window: pd.DataFrame,
    location: Location,
    ac_capacity_kw: float,
    project_root: Path,
) -> dict[str, Any]:
    perfect = perfect_information_baseline(window, protocol.battery)
    mpc = rolling_horizon_baseline(window, protocol.battery, lookahead_steps=96)
    event_results, event_artifacts = _event_results(protocol, window, perfect.action_kw, ac_capacity_kw)
    no_action_cost = _no_action_cost(window, protocol)
    checks = {
        "night_pv_near_zero": _night_pv_near_zero(window, location),
        "pv_within_nameplate": bool(
            (window["pv_ac_kw"] >= -1e-8).all() and (window["pv_ac_kw"] <= ac_capacity_kw + 1e-6).all()
        ),
        "battery_energy_conservation": _energy_conserved(perfect.simulation.trace, protocol),
        "soc_and_power_constraints": _battery_constraints(perfect.simulation.trace, protocol),
        "oracle_plan_feasible": perfect.simulation.feasible and all(item["oracle_feasible"] for item in event_results),
        "event_material_impact": all(item["material_impact"] for item in event_results),
        "wrong_state_rejected": all(item["wrong_state_rejected"] for item in event_results),
        "scorer_ceiling": perfect.simulation.total_cost <= no_action_cost + 1e-5
        and all(item["correct_state_reaches_ceiling"] for item in event_results),
        "perfect_not_unexplained_worse_than_mpc": perfect.simulation.total_cost <= mpc.simulation.total_cost + 1e-5,
        "no_llm_call_records": _no_llm_call_records(project_root),
    }
    return {
        "window_kind": kind,
        "start_timestamp_utc": window.index[0].isoformat(),
        "end_timestamp_utc": window.index[-1].isoformat(),
        "intervals": len(window),
        "perfect_information_cost": perfect.simulation.total_cost,
        "rolling_horizon_mpc_cost": mpc.simulation.total_cost,
        "scorer_ceiling_no_action_cost": no_action_cost,
        "event_results": event_results,
        "diagnostic_plots": _write_diagnostic_plots(
            project_root, kind, window, perfect.simulation.trace, mpc.simulation.trace, event_artifacts
        )
        if kind == "stress"
        else [],
        "checks": checks,
    }


def _event_results(
    protocol: FrozenProtocol, window: pd.DataFrame, nominal_actions: np.ndarray, ac_capacity_kw: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    forecast_planning = window.copy()
    forecast_planning["pv_ac_kw"] = forecast_planning["forecast_pv_kw"]
    for offset, family in enumerate(protocol.scenario.event_families):
        event = generate_event(family, "hard", protocol.selection.event_generation_seed + offset, protocol.battery)
        schedule = dynamic_override_schedule(event.visible_update, len(window))
        if family == "forecast_revision":
            activation = int(event.visible_update["activation_step"])
            before = perfect_information_baseline(forecast_planning, protocol.battery)
            revised = forecast_planning.copy()
            revised["pv_ac_kw"] = revised_pv_forecast(
                window,
                activation,
                float(event.visible_update["forecast_revision_blend_weight"]),
                ac_capacity_kw,
            )
            visible_plan = perfect_information_baseline(revised, protocol.battery)
            oracle_plan = perfect_information_baseline(window, protocol.battery)
            executed = simulate_dispatch(window, visible_plan.action_kw, protocol.battery)
            wrong_state = simulate_dispatch(window, before.action_kw, protocol.battery)
            action_distance = float(np.abs(visible_plan.action_kw - before.action_kw).sum())
        else:
            visible_plan = perfect_information_baseline(window, protocol.battery, overrides_by_step=schedule)
            oracle_plan = visible_plan
            executed = visible_plan.simulation
            wrong_state = simulate_dispatch(window, nominal_actions, protocol.battery, overrides_by_step=schedule)
            action_distance = float(np.abs(visible_plan.action_kw - nominal_actions).sum())
        cost_distance = abs(executed.total_cost - oracle_plan.simulation.total_cost)
        wrong_state_rejected = bool(
            not wrong_state.feasible
            or wrong_state.total_cost > oracle_plan.simulation.total_cost + 1e-4
            or wrong_state.trace["violation_cost"].sum() > 1e-8
            or action_distance > 1e-3
        )
        results.append(
            {
                "event_family": family,
                "difficulty": "hard",
                "activation_step": event.visible_update["activation_step"],
                "oracle_feasible": oracle_plan.simulation.feasible,
                "executed_feasible": executed.feasible,
                "action_l1_distance_kw": action_distance,
                "cost_distance_usd": cost_distance,
                "material_impact": action_distance > 1e-3 or cost_distance > 1e-4,
                "wrong_state_rejected": wrong_state_rejected,
                "correct_state_reaches_ceiling": oracle_plan.simulation.feasible
                and oracle_plan.simulation.total_cost <= wrong_state.total_cost + 1e-5,
            }
        )
        artifacts.append(
            {
                "event_family": family,
                "visible_update": event.visible_update,
                "baseline_action_kw": nominal_actions,
                "updated_action_kw": visible_plan.action_kw,
                "constraint_trace": executed.trace,
            }
        )
    return results, artifacts


def _write_diagnostic_plots(
    project_root: Path,
    kind: str,
    window: pd.DataFrame,
    perfect_trace: pd.DataFrame,
    mpc_trace: pd.DataFrame,
    events: list[dict[str, Any]],
) -> list[str]:
    """Write the four required real-data diagnostic figures for the fixed stress window."""
    destination = project_root / "reports" / "smoke_diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "pv_irradiance": destination / "pv_and_irradiance.png",
        "pv_load_price": destination / "pv_load_price.png",
        "soc": destination / "perfect_information_mpc_soc.png",
        "events": destination / "dynamic_event_constraint_action_changes.png",
    }
    _plot_pv_irradiance(window, paths["pv_irradiance"])
    _plot_pv_load_price(window, paths["pv_load_price"])
    _plot_soc(perfect_trace, mpc_trace, paths["soc"])
    _plot_events(events, paths["events"])
    return [str(path) for path in paths.values()]


def _plot_pv_irradiance(window: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(window.index, window["ghi"], color="tab:orange", label="GHI (W/m²)")
    axis.plot(window.index, window["dni"], color="tab:red", alpha=0.7, label="DNI (W/m²)")
    pv_axis = axis.twinx()
    pv_axis.plot(window.index, window["pv_ac_kw"], color="tab:blue", label="PV AC (kW)")
    axis.set_ylabel("Irradiance (W/m²)")
    pv_axis.set_ylabel("PV AC (kW)")
    _legend(axis, pv_axis)
    _save_figure(figure, path, "Stress-window PV AC and irradiance")


def _plot_pv_load_price(window: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(window.index, window["pv_ac_kw"], color="tab:blue", label="PV AC (kW)")
    axis.plot(window.index, window["load_kw"], color="tab:green", label="Load (kW)")
    price_axis = axis.twinx()
    price_axis.plot(
        window.index, window["day_ahead_price_usd_mwh"], color="tab:purple", alpha=0.75, label="DAM price (USD/MWh)"
    )
    axis.set_ylabel("Power (kW)")
    price_axis.set_ylabel("Price (USD/MWh)")
    _legend(axis, price_axis)
    _save_figure(figure, path, "Stress-window PV, load, and day-ahead price")


def _plot_soc(perfect_trace: pd.DataFrame, mpc_trace: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(perfect_trace.index, perfect_trace["soc_kwh"], color="tab:blue", label="Perfect-information SOC")
    axis.plot(mpc_trace.index, mpc_trace["soc_kwh"], color="tab:orange", label="Rolling-horizon MPC SOC")
    axis.plot(perfect_trace.index, perfect_trace["reserve_soc_kwh"], color="black", linestyle="--", label="Reserve SOC")
    axis.set_ylabel("SOC (kWh)")
    axis.legend(loc="best")
    _save_figure(figure, path, "Perfect-information and MPC state of charge")


def _plot_events(events: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(len(events), 1, figsize=(11, 2.2 * len(events)), sharex=True)
    for axis, artifact in zip(np.atleast_1d(axes), events, strict=True):
        trace = cast(pd.DataFrame, artifact["constraint_trace"])
        axis.plot(trace.index, artifact["baseline_action_kw"], color="0.55", label="Pre-event action")
        axis.plot(trace.index, artifact["updated_action_kw"], color="tab:blue", label="Updated action")
        activation = int(cast(dict[str, Any], artifact["visible_update"])["activation_step"])
        axis.axvline(trace.index[activation], color="tab:red", linestyle="--", linewidth=1)
        axis.plot(trace.index, trace["available_capacity_kwh"], color="tab:green", alpha=0.65, label="Available capacity")
        axis.plot(trace.index, trace["reserve_soc_kwh"], color="tab:orange", alpha=0.65, label="Reserve SOC")
        axis.set_title(str(artifact["event_family"]))
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=2, fontsize=8)
    _save_figure(figure, path, "Dynamic-event constraints and actions before/after activation")


def _legend(left: Any, right: Any) -> None:
    handles, labels = left.get_legend_handles_labels()
    right_handles, right_labels = right.get_legend_handles_labels()
    left.legend(handles + right_handles, labels + right_labels, loc="best")


def _save_figure(figure: Any, path: Path, title: str) -> None:
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _night_pv_near_zero(window: pd.DataFrame, location: Location) -> bool:
    solar = pvlib.solarposition.get_solarposition(
        window.index, location.latitude, location.longitude, altitude=location.altitude_m
    )
    night = window.loc[solar["apparent_zenith"] >= 90, "pv_ac_kw"]
    return bool(not night.empty and night.abs().max() <= 0.5)


def _energy_conserved(trace: pd.DataFrame, protocol: FrozenProtocol) -> bool:
    dt = 0.25
    previous = np.concatenate(([protocol.battery.initial_soc_kwh], trace["soc_kwh"].to_numpy()[:-1]))
    expected = previous + trace["charge_kw"].to_numpy() * protocol.battery.charge_efficiency * dt
    expected -= trace["discharge_kw"].to_numpy() / protocol.battery.discharge_efficiency * dt
    return bool(np.max(np.abs(expected - trace["soc_kwh"].to_numpy())) <= 1e-6)


def _battery_constraints(trace: pd.DataFrame, protocol: FrozenProtocol) -> bool:
    return bool(
        (trace["soc_kwh"] >= trace["reserve_soc_kwh"] - 1e-8).all()
        and (trace["soc_kwh"] <= trace["available_capacity_kwh"] + 1e-8).all()
        and (trace["charge_kw"] <= protocol.battery.max_charge_kw + 1e-8).all()
        and (trace["discharge_kw"] <= protocol.battery.max_discharge_kw + 1e-8).all()
    )


def _no_action_cost(window: pd.DataFrame, protocol: FrozenProtocol) -> float:
    return simulate_dispatch(window, np.zeros(len(window)), protocol.battery).total_cost


def _location(protocol: FrozenProtocol, site: str) -> Location:
    for location in protocol.locations:
        if location.location_id == site:
            return location
    raise ValueError(f"Unknown site: {site}")


def _no_llm_call_records(project_root: Path) -> bool:
    logs = project_root / "logs"
    candidates = list(logs.glob("*llm*")) + list(logs.glob("*model*call*")) if logs.exists() else []
    return all(path.stat().st_size == 0 for path in candidates if path.is_file())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _blocked_report(protocol: FrozenProtocol, site: str, reason: str) -> dict[str, Any]:
    return {
        "report_type": "DETERMINISTIC_SMOKE_REPORT",
        "created_utc": utc_now(),
        "status": "blocked",
        "site": site,
        "protocol_version": protocol.protocol_version,
        "reason": reason,
        "passed": False,
    }
