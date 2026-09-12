# SEGAN offline recomputation package

This package was generated entirely from saved F1/F0 tasks and responses. It performs no provider or network calls.

## Main files

- `OFFLINE_RECOMPUTE_REPORT.md`: concise result narrative.
- `OFFLINE_RECOMPUTE_SUMMARY.json`: complete machine-readable summary.
- `f1_branch_metrics.csv`, `f1_block_metrics.csv`: primary recomputed metrics.
- `f1_stratified_summary.csv`: SOC gap, direction, event, horizon, and other strata.
- `f1_correlation_summary.csv`: dose, horizon, timing, and latency associations.
- `f1_gate_branch_results.csv`: F1 branch-level same-space and expanded-system gates.
- `gate_baseline_results.csv`: zero, heuristic, stale-optimal, random, and direct-MPC baselines.
- `gate_action_sidecars.jsonl.gz`: full gated action, curtailment, and SOC vectors.
- `independent_replay_comparison.csv`: independent versus archived replay checks.
- `f1_f0_branch_pairs.csv`, `f1_f0_block_pairs.csv`: exact-payload temporal pairing.
- `recompute_offline.py`: full regeneration script.

Bootstrap intervals are scenario-resampling stability intervals for the frozen challenge set.
