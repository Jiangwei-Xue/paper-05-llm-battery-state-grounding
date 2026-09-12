"""Deterministic, row-evidence-only analyses for the EnergyBench release."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

BINARY_METRICS = (
    "G_state_governance_success",
    "A_governed_action_success",
    "D_dispatch_execution_success",
    "C_feasible_operational_feasibility_success",
    "C_economic_operational_economic_success",
    "J_feasible",
    "J_economic",
)

METHOD_AXES = {
    "rolling_summary": ("natural_language", "model"),
    "visible_carry": ("natural_language", "deterministic_system"),
    "typed_state": ("typed_json", "model"),
    "canonical_typed_carry": ("typed_json", "deterministic_system"),
}


def binary_rates(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    denominator = len(records)
    return {
        metric: {
            "successes": sum(record["outcome"].get(metric) is True for record in records),
            "denominator": denominator,
            "rate": (
                sum(record["outcome"].get(metric) is True for record in records) / denominator
                if denominator
                else 0.0
            ),
        }
        for metric in BINARY_METRICS
    }


def stratified_rates(records: list[dict[str, Any]], *fields: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record[field]) for field in fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key, subset in sorted(grouped.items()):
        row: dict[str, Any] = {field: value for field, value in zip(fields, key, strict=True)}
        row["episodes"] = len(subset)
        row["parser_successes"] = sum(record["outcome"].get("parser_success") is True for record in subset)
        row["metrics"] = binary_rates(subset)
        rows.append(row)
    return rows


def joint_quadrants(records: list[dict[str, Any]], c_metric: str) -> dict[str, dict[str, float | int]]:
    labels = ("G=1,C=1", "G=1,C=0", "G=0,C=1", "G=0,C=0")
    counts = Counter(
        f"G={int(record['outcome'].get('G_state_governance_success') is True)},"
        f"C={int(record['outcome'].get(c_metric) is True)}"
        for record in records
    )
    total = len(records)
    return {
        label: {"count": counts[label], "denominator": total, "rate": counts[label] / total if total else 0.0}
        for label in labels
    }


def factorial_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for method, (schema, owner) in METHOD_AXES.items():
        subset = [record for record in records if record["state_method"] == method]
        cells[method] = {
            "state_schema": schema,
            "state_owner": owner,
            "episodes": len(subset),
            "metrics": binary_rates(subset),
            "by_provider": stratified_rates(subset, "provider"),
        }

    def rate(method: str, metric: str, subset: list[dict[str, Any]] | None = None) -> float:
        source = records if subset is None else subset
        selected = [record for record in source if record["state_method"] == method]
        return float(np.mean([record["outcome"].get(metric) is True for record in selected]))

    def contrasts_for(source: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for metric in BINARY_METRICS:
            rolling = rate("rolling_summary", metric, source)
            visible = rate("visible_carry", metric, source)
            typed = rate("typed_state", metric, source)
            canonical = rate("canonical_typed_carry", metric, source)
            output[metric] = {
                "state_schema_typed_minus_natural": ((typed + canonical) - (rolling + visible)) / 2,
                "state_owner_system_minus_model": ((visible + canonical) - (rolling + typed)) / 2,
                "schema_by_owner_interaction": (canonical - typed) - (visible - rolling),
            }
        return output

    providers = sorted({str(record["provider"]) for record in records})
    return {
        "design": "2x2_state_schema_by_state_owner",
        "cells": cells,
        "contrasts": contrasts_for(records),
        "contrasts_by_provider": {
            provider: contrasts_for([record for record in records if record["provider"] == provider])
            for provider in providers
        },
    }


def repeatability_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["scenario_id"], record["provider"], record["state_method"])].append(record)
    for subset in grouped.values():
        subset.sort(key=lambda item: int(item["repetition"]))
    if any(len(subset) != 3 for subset in grouped.values()):
        raise ValueError("Every scenario-provider-method cell must contain exactly three repetitions.")

    metrics: dict[str, Any] = {}
    for metric in BINARY_METRICS:
        counts = Counter(sum(record["outcome"].get(metric) is True for record in subset) for subset in grouped.values())
        metric_at_k = {}
        for k in (1, 2, 3):
            successes = sum(
                any(record["outcome"].get(metric) is True for record in subset[:k])
                for subset in grouped.values()
            )
            metric_at_k[str(k)] = {
                "successful_cells": successes,
                "denominator_cells": len(grouped),
                "rate": successes / len(grouped),
            }
        metrics[metric] = {
            "metric_at_k": metric_at_k,
            "success_count_distribution": {
                str(count): {
                    "cells": counts[count],
                    "denominator_cells": len(grouped),
                    "rate": counts[count] / len(grouped),
                }
                for count in range(4)
            },
            "within_cell_disagreement": {
                "cells": counts[1] + counts[2],
                "denominator_cells": len(grouped),
                "rate": (counts[1] + counts[2]) / len(grouped),
            },
            "stable_zero": counts[0],
            "stable_positive": counts[3],
        }
    return {"cell_definition": "scenario_x_provider_x_method", "cell_count": len(grouped), "metrics": metrics}


def support_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_ids = sorted({record["scenario_id"] for record in records})

    def success_support(field: str, metric: str) -> dict[str, Any]:
        values = sorted({str(record[field]) for record in records})
        return {
            value: {
                "episodes": sum(str(record[field]) == value for record in records),
                "successful_episodes": sum(
                    str(record[field]) == value and record["outcome"].get(metric) is True for record in records
                ),
                "successful_scenarios": len(
                    {
                        record["scenario_id"]
                        for record in records
                        if str(record[field]) == value and record["outcome"].get(metric) is True
                    }
                ),
            }
            for value in values
        }

    def leave_one_out(field: str) -> list[dict[str, Any]]:
        return [
            {
                "left_out": value,
                "episodes": len(subset := [record for record in records if str(record[field]) != value]),
                "scenario_support": len({record["scenario_id"] for record in subset}),
                "metrics": binary_rates(subset),
            }
            for value in sorted({str(record[field]) for record in records})
        ]

    return {
        "episode_count": len(records),
        "scenario_count": len(scenario_ids),
        "independence_warning": "Episodes are repeated measurements nested within scenarios, not independent energy systems.",
        "successful_scenarios": {
            metric: len(
                {
                    record["scenario_id"]
                    for record in records
                    if record["outcome"].get(metric) is True
                }
            )
            for metric in BINARY_METRICS
        },
        "event_family_support": success_support("event_family", "C_economic_operational_economic_success"),
        "site_support": success_support("location_id", "C_economic_operational_economic_success"),
        "season_support": success_support("season", "C_economic_operational_economic_success"),
        "leave_one_site_out": leave_one_out("location_id"),
        "leave_one_event_family_out": leave_one_out("event_family"),
    }


def failure_accounting(records: list[dict[str, Any]]) -> dict[str, Any]:
    protocol_exclusions = Counter(
        code
        for record in records
        for code in record.get("failure_classification", {}).get("protocol_exclusion_codes", [])
    )
    transport = Counter(
        str(record[stage]["response"].get("status"))
        for record in records
        for stage in ("stage1", "stage2")
    )
    parser_success = [record for record in records if record["outcome"].get("parser_success") is True]
    return {
        "episodes": len(records),
        "failure_as_zero_primary": binary_rates(records),
        "parse_success_only_sensitivity_nonprimary": {
            "episodes": len(parser_success),
            "excluded_parser_failures": len(records) - len(parser_success),
            "metrics": binary_rates(parser_success),
        },
        "parser_successes": len(parser_success),
        "parser_failures": len(records) - len(parser_success),
        "transport_stage_status_counts": dict(sorted(transport.items())),
        "failure_codes": dict(
            sorted(Counter(code for record in records for code in record["outcome"].get("failure_codes", [])).items())
        ),
        "constraint_failure_codes": dict(
            sorted(
                Counter(
                    code
                    for record in records
                    for code in record["outcome"].get("constraint_failure_codes", [])
                ).items()
            )
        ),
        "protocol_exclusions": dict(sorted(protocol_exclusions.items())),
        "manual_repairs": 0,
        "complete_case_filtering": False,
    }


def cluster_bootstrap(
    records: list[dict[str, Any]], *, seed: int = 20260716, resamples: int = 10_000
) -> dict[str, Any]:
    scenarios = sorted({record["scenario_id"] for record in records})
    by_scenario = {scenario: [record for record in records if record["scenario_id"] == scenario] for scenario in scenarios}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(scenarios), size=(resamples, len(scenarios)))

    def interval(values: np.ndarray, point: float) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {"estimate": point, "ci95_low": float(low), "ci95_high": float(high)}

    rates: dict[str, Any] = {}
    for metric in BINARY_METRICS:
        scenario_success_counts = np.asarray(
            [sum(record["outcome"].get(metric) is True for record in by_scenario[scenario]) for scenario in scenarios]
        )
        per_scenario = np.asarray(
            [np.mean([record["outcome"].get(metric) is True for record in by_scenario[scenario]]) for scenario in scenarios]
        )
        draws = per_scenario[sampled].mean(axis=1)
        rates[metric] = {
            **interval(draws, float(per_scenario.mean())),
            "denominator_episodes": len(records),
            "scenario_support": len(scenarios),
            "zero_mass": float(np.mean(draws == 0.0)),
            "one_mass": float(np.mean(draws == 1.0)),
            "zero_success_scenarios": int(np.sum(scenario_success_counts == 0)),
            "max_scenario_success_share": (
                float(scenario_success_counts.max() / scenario_success_counts.sum())
                if scenario_success_counts.sum()
                else None
            ),
        }

    scenario_method: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in scenarios:
        scenario_method[scenario] = {}
        for method in METHOD_AXES:
            subset = [record for record in by_scenario[scenario] if record["state_method"] == method]
            scenario_method[scenario][method] = {
                metric: float(np.mean([record["outcome"].get(metric) is True for record in subset]))
                for metric in BINARY_METRICS
            }

    contrasts: dict[str, Any] = {}
    method_contrasts: dict[str, Any] = {}
    for metric in BINARY_METRICS:
        vectors: dict[str, np.ndarray] = {
            method: np.asarray([scenario_method[scenario][method][metric] for scenario in scenarios])
            for method in METHOD_AXES
        }
        contrast_vectors = {
            "state_schema_typed_minus_natural": (
                vectors["typed_state"]
                + vectors["canonical_typed_carry"]
                - vectors["rolling_summary"]
                - vectors["visible_carry"]
            )
            / 2,
            "state_owner_system_minus_model": (
                vectors["visible_carry"]
                + vectors["canonical_typed_carry"]
                - vectors["rolling_summary"]
                - vectors["typed_state"]
            )
            / 2,
            "schema_by_owner_interaction": (
                vectors["canonical_typed_carry"]
                - vectors["typed_state"]
                - vectors["visible_carry"]
                + vectors["rolling_summary"]
            ),
        }
        contrasts[metric] = {
            name: {
                **interval(vector[sampled].mean(axis=1), float(vector.mean())),
                "paired_by_scenario": True,
                "scenario_support": len(scenarios),
            }
            for name, vector in contrast_vectors.items()
        }
        method_pairs: dict[str, Any] = {}
        methods = tuple(METHOD_AXES)
        for left_index, left in enumerate(methods):
            for right in methods[left_index + 1 :]:
                vector = vectors[left] - vectors[right]
                method_pairs[f"{left}_minus_{right}"] = {
                    **interval(vector[sampled].mean(axis=1), float(vector.mean())),
                    "paired_by_scenario": True,
                    "scenario_support": len(scenarios),
                }
        method_contrasts[metric] = method_pairs
    return {
        "seed": seed,
        "resamples": resamples,
        "cluster_unit": "scenario",
        "tier_pooled_with_other_tiers": False,
        "headline_rates": rates,
        "paired_method_contrasts": method_contrasts,
        "paired_factorial_contrasts": contrasts,
    }


def safety_shield_analysis(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    applicable = [row for row in rows if row.get("applicable") is True]
    projected = [row for row in applicable if "shielded_feasible" in row]
    raw_infeasible = [row for row in projected if row.get("raw_feasible") is False]
    repaired = [row for row in raw_infeasible if row.get("shielded_feasible") is True]
    costs = [
        float(row["shielded_total_cost"]) - float(row["raw_total_cost"])
        for row in projected
        if row.get("raw_total_cost") is not None and row.get("shielded_total_cost") is not None
    ]
    state_failures = {record["run_id"] for record in records if record["outcome"].get("G_state_governance_success") is False}
    return {
        "sidecar_only": True,
        "raw_primary_results_replaced": False,
        "episodes": len(rows),
        "applicable_parser_success_rows": len(applicable),
        "raw_feasible_rows": sum(row.get("raw_feasible") is True for row in projected),
        "shielded_feasible_rows": sum(row.get("shielded_feasible") is True for row in projected),
        "raw_infeasible_rows_with_projection_result": len(raw_infeasible),
        "constraint_violation_repairs": len(repaired),
        "constraint_violation_repair_fraction": len(repaired) / len(raw_infeasible) if raw_infeasible else None,
        "mean_l1_correction_kw": (
            float(np.mean([float(row.get("l1_action_adjustment_kw", 0.0)) for row in applicable]))
            if applicable
            else None
        ),
        "median_l1_correction_kw": (
            float(np.median([float(row.get("l1_action_adjustment_kw", 0.0)) for row in applicable]))
            if applicable
            else None
        ),
        "mean_shield_minus_raw_cost": float(np.mean(costs)) if costs else None,
        "state_failure_rows": len(state_failures),
        "state_failures_repaired_by_shield": 0,
        "boundary": "The shield projects actions only; it does not alter or repair model state outputs.",
        "classification_counts": dict(sorted(Counter(str(row.get("classification")) for row in rows).items())),
    }


def method_direction_comparison(primary: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for metric in BINARY_METRICS:
        for contrast in (
            "state_schema_typed_minus_natural",
            "state_owner_system_minus_model",
            "schema_by_owner_interaction",
        ):
            p = primary["contrasts"][metric][contrast]
            e = extension["contrasts"][metric][contrast]
            rows.append(
                {
                    "metric": metric,
                    "contrast": contrast,
                    "primary": p,
                    "extension": e,
                    "same_direction": bool(np.sign(p) == np.sign(e)),
                }
            )
    return {
        "rows": rows,
        "same_direction_count": sum(row["same_direction"] for row in rows),
        "comparison_count": len(rows),
        "tiers_pooled": False,
    }


def row_ids(records: Iterable[dict[str, Any]], predicate: Any | None = None) -> list[str]:
    if predicate is None:
        return sorted(str(record["run_id"]) for record in records)
    return sorted(str(record["run_id"]) for record in records if predicate(record))
