"""Deterministic analyses for the strict paired-authority corrective tier."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from energy_agent_reliability.final_analysis import (
    BINARY_METRICS,
    binary_rates,
    cluster_bootstrap,
    factorial_analysis,
    joint_quadrants,
    support_analysis,
)

PAIR_METRICS = ("parser_success", *BINARY_METRICS)
REPRESENTATIONS = ("natural_language", "typed_json")
OWNERS = ("model", "deterministic_system")


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose the four scoring methods expected by the registered 2x2 analysis."""
    normalized = []
    for record in records:
        row = dict(record)
        row["state_method"] = record["scoring_state_method"]
        normalized.append(row)
    return normalized


def validate_corrective_inputs(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the frozen 720-pair/1,440-branch matrix without changing rows."""
    record_ids = [str(record["run_id"]) for record in records]
    pair_ids = [str(pair["paired_run_id"]) for pair in pairs]
    plan_ids = [str(row["paired_run_id"]) for row in plan]
    records_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_pair[str(record["paired_run_id"])].append(record)

    expected_cell_counts = {"primary": 40, "extension": 20}
    cell_counts = Counter(
        (
            str(record["corrective_evidence_tier"]),
            str(record["provider"]),
            str(record["representation"]),
            str(record["state_owner"]),
            int(record["repetition"]),
        )
        for record in records
    )
    matrix_pass = all(
        count == expected_cell_counts[tier]
        for (tier, _provider, _representation, _owner, _repetition), count in cell_counts.items()
    ) and len(cell_counts) == 2 * 2 * 2 * 2 * 3

    gates = {
        "pair_count_720": len(pairs) == 720,
        "branch_count_1440": len(records) == 1440,
        "plan_count_720": len(plan) == 720,
        "unique_pair_ids": len(set(pair_ids)) == len(pair_ids),
        "unique_branch_ids": len(set(record_ids)) == len(record_ids),
        "plan_pair_ids_match_records": set(plan_ids) == set(pair_ids),
        "two_owner_branches_per_pair": all(
            len(subset) == 2
            and {str(record["state_owner"]) for record in subset} == set(OWNERS)
            for subset in records_by_pair.values()
        ),
        "matrix_exact": matrix_pass,
        "pair_invariants_pass": all(
            all(bool(value) for value in pair["pair_invariants"].values())
            for pair in pairs
        ),
        "protocol_exclusions_zero": not any(
            pair.get("protocol_exclusion_codes") for pair in pairs
        )
        and not any(
            record.get("failure_classification", {}).get(
                "protocol_exclusion_codes"
            )
            for record in records
        ),
        "excluded_flag_false": all(
            record.get("excluded_from_formal_analysis") is False
            for record in records
        ),
        "record_protocol_id_matches": {
            str(record["protocol_id"]) for record in records
        }
        == {"energybench_paired_authority_corrective_v1"},
    }
    eligibility = Counter(
        bool(record.get("formal_analysis_eligible")) for record in records
    )
    metadata_conflict = (
        gates["excluded_flag_false"]
        and gates["protocol_exclusions_zero"]
        and eligibility[False] == len(records)
    )
    return {
        "gates": gates,
        "all_structural_gates_pass": all(gates.values()),
        "formal_analysis_eligible_counts": {
            "true": eligibility[True],
            "false": eligibility[False],
        },
        "stale_runner_metadata_conflict": metadata_conflict,
        "claim_use_gate": "hold" if metadata_conflict else "pass",
        "metadata_boundary": (
            "The runner hard-coded formal_analysis_eligible=false while the "
            "prospective corrective protocol set claim_bearing=true and every "
            "row set excluded_from_formal_analysis=false. The analysis preserves "
            "all row-level files and reports this conflict instead of rewriting them."
        ),
    }


def analyze_tier_provider(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    tier: str,
    provider: str,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    subset = [
        record
        for record in records
        if record["corrective_evidence_tier"] == tier
        and record["provider"] == provider
    ]
    pair_subset = [
        pair
        for pair in pairs
        if pair["corrective_evidence_tier"] == tier
        and pair["provider"] == provider
    ]
    normalized = normalize_records(subset)
    return {
        "tier": tier,
        "provider": provider,
        "branch_records": len(subset),
        "paired_runs": len(pair_subset),
        "scenario_count": len({str(record["scenario_id"]) for record in subset}),
        "failure_as_zero_rates": binary_rates(normalized),
        "joint_quadrants_economic": joint_quadrants(
            normalized, "C_economic_operational_economic_success"
        ),
        "cells": factorial_analysis(normalized)["cells"],
        "factorial_point_contrasts": factorial_analysis(normalized)["contrasts"],
        "scenario_cluster_bootstrap": cluster_bootstrap(
            normalized, seed=seed, resamples=resamples
        ),
        "strict_owner_pairs": strict_owner_pair_analysis(
            subset,
            pair_subset,
            seed=seed,
            resamples=resamples,
        ),
        "support": support_analysis(normalized),
    }


def strict_owner_pair_analysis(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Estimate system-minus-model differences from the two branches of each pair."""
    by_run_id = {str(record["run_id"]): record for record in records}
    output: dict[str, Any] = {}
    scenario_differences: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for representation in REPRESENTATIONS:
        rep_pairs = [
            pair for pair in pairs if pair["representation"] == representation
        ]
        metric_rows: dict[str, Any] = {}
        for metric in PAIR_METRICS:
            rows = []
            for pair in rep_pairs:
                model = by_run_id[str(pair["branch_record_ids"]["model"])]
                system = by_run_id[
                    str(pair["branch_record_ids"]["deterministic_system"])
                ]
                model_value = model["outcome"].get(metric) is True
                system_value = system["outcome"].get(metric) is True
                rows.append(
                    {
                        "paired_run_id": pair["paired_run_id"],
                        "scenario_id": pair["scenario_id"],
                        "repetition": int(pair["repetition"]),
                        "model": model_value,
                        "deterministic_system": system_value,
                        "difference": int(system_value) - int(model_value),
                    }
                )
            interval = scenario_cluster_pair_interval(
                rows,
                seed=seed,
                resamples=resamples,
            )
            discordance = Counter(
                {
                    (True, True): "both_success",
                    (False, True): "system_only",
                    (True, False): "model_only",
                    (False, False): "neither_success",
                }[(row["model"], row["deterministic_system"])]
                for row in rows
            )
            metric_rows[metric] = {
                "pair_count": len(rows),
                "scenario_count": len({str(row["scenario_id"]) for row in rows}),
                "model_successes": sum(row["model"] for row in rows),
                "system_successes": sum(
                    row["deterministic_system"] for row in rows
                ),
                "discordance": {
                    label: discordance[label]
                    for label in (
                        "both_success",
                        "system_only",
                        "model_only",
                        "neither_success",
                    )
                },
                "system_minus_model": interval,
                "repetition_difference_counts": {
                    str(repetition): dict(
                        sorted(
                            Counter(
                                int(row["difference"])
                                for row in rows
                                if int(row["repetition"]) == repetition
                            ).items()
                        )
                    )
                    for repetition in range(3)
                },
            }
            for scenario, value in interval["scenario_mean_differences"].items():
                scenario_differences[scenario][representation][metric] = value
        output[representation] = metric_rows

    interactions: dict[str, Any] = {}
    for metric in PAIR_METRICS:
        scenarios = sorted(
            scenario
            for scenario, representations in scenario_differences.items()
            if set(representations) == set(REPRESENTATIONS)
        )
        vector = np.asarray(
            [
                scenario_differences[scenario]["typed_json"][metric]
                - scenario_differences[scenario]["natural_language"][metric]
                for scenario in scenarios
            ],
            dtype=float,
        )
        interactions[metric] = vector_interval(
            vector,
            seed=seed,
            resamples=resamples,
            scenario_count=len(scenarios),
        )
        interactions[metric]["definition"] = (
            "(system-model typed_json) - (system-model natural_language)"
        )
    return {
        "pairing_unit": (
            "shared Stage-1 response, parsed dispatch, executed prefix, event state, "
            "and Stage-2 non-carrier prompt surface"
        ),
        "representations": output,
        "owner_by_representation_interaction": interactions,
    }


def scenario_cluster_pair_interval(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(float(row["difference"]))
    scenarios = sorted(grouped)
    means = np.asarray(
        [float(np.mean(grouped[scenario])) for scenario in scenarios], dtype=float
    )
    result = vector_interval(
        means,
        seed=seed,
        resamples=resamples,
        scenario_count=len(scenarios),
    )
    result["scenario_mean_differences"] = {
        scenario: float(value) for scenario, value in zip(scenarios, means, strict=True)
    }
    return result


def vector_interval(
    vector: np.ndarray,
    *,
    seed: int,
    resamples: int,
    scenario_count: int,
) -> dict[str, Any]:
    if len(vector) == 0:
        raise ValueError("Cannot bootstrap an empty scenario vector.")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(vector), size=(resamples, len(vector)))
    draws = vector[sampled].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(vector.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "scenario_count": scenario_count,
        "cluster_unit": "scenario",
        "resamples": resamples,
        "bootstrap_seed": seed,
        "zero_mass": float(np.mean(draws == 0.0)),
    }


def direction_comparison(
    analyses: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    rows = []
    for provider in sorted(analyses["primary"]):
        primary = analyses["primary"][provider][
            "scenario_cluster_bootstrap"
        ]["paired_factorial_contrasts"]
        extension = analyses["extension"][provider][
            "scenario_cluster_bootstrap"
        ]["paired_factorial_contrasts"]
        for metric in BINARY_METRICS:
            for contrast in (
                "state_schema_typed_minus_natural",
                "state_owner_system_minus_model",
                "schema_by_owner_interaction",
            ):
                p = float(primary[metric][contrast]["estimate"])
                e = float(extension[metric][contrast]["estimate"])
                rows.append(
                    {
                        "provider": provider,
                        "metric": metric,
                        "contrast": contrast,
                        "primary_estimate": p,
                        "extension_estimate": e,
                        "same_direction": bool(np.sign(p) == np.sign(e)),
                    }
                )
    return {
        "rows": rows,
        "same_direction": sum(row["same_direction"] for row in rows),
        "comparisons": len(rows),
        "models_pooled": False,
        "tiers_pooled": False,
    }
