# Paper results map

The bundled paper contains nine tables and no figure environments. This map
links each table to retained row-level evidence and its regeneration path.

| Paper item | Scientific content | Denominator | Primary evidence | Reproduction stage |
|---|---|---:|---|---|
| `tab:tiers` | experiment tiers and scopes | 825 records | `experiments_manifest.json`; `saved_records/INDEX.json` | release-contract audit |
| `tab:e2-core` | E2 canonical-stale and duplicate-null distances | 60 selected scenarios | `reference_outputs/primary/OFFLINE_RECOMPUTE_SUMMARY.json`; `f1_block_metrics.csv` | primary recomputation |
| `tab:f1-phenotypes` | activity and violation mechanisms by model | 480 F1 branches | `f1_branch_metrics.csv`; `OFFLINE_COMPLETION_AUDIT.json`; `UNIFIED_VIOLATION_SEVERITY.jsonl` | primary + independent physics |
| `tab:dose` | response by 50/75/100 kWh state gap | 60 scenarios by model tier | `f1_block_metrics.csv`; `OFFLINE_OPTIMIZATION_SENSITIVITIES.json` | sensitivity analysis |
| `tab:dual-replay` | represented-state and authoritative-state feasibility | F1 480, F0 120, QWEN37 300 | `FEASIBILITY_SEMANTICS.json`; `table_feasibility_by_replay_state.tex` | feasibility semantics |
| `tab:gate-zero` | model proposal versus zero proposal under one gate | 480 F1 branches plus matched baselines | `f1_gate_branch_results.csv`; `gate_baseline_results.csv` | primary gate recomputation |
| `tab:event` | event-family stratification | retained F1 scenarios | `f1_block_metrics.csv`; `OFFLINE_COMPLETION_AUDIT.json` | primary + audit |
| `tab:witnesses` | represented-feasible but authoritative-infeasible cases | QWEN37 F1/F0 branches | `FEASIBILITY_SEMANTICS.json`; `QWEN37_F1_BRANCH_METRICS.csv` | QWEN37 + semantics |
| `tab:baselines` | zero, heuristic, random, and MPC comparisons | 300 matched scenario-baseline rows | `gate_baseline_results.csv`; `OFFLINE_RECOMPUTE_SUMMARY.json` | matched baseline recomputation |

The more detailed claim-level registry is `CLAIM_TO_EVIDENCE_MAP.json`, and the
machine-readable table registry is `results_manifest.json`.

## Interpretation controls

- Historical Flash conditions and Qwen3.7-Plus are never pooled.
- Carrier-consistent and authoritative feasibility are never reported under an
  unqualified `raw feasibility` label.
- E2 raw-action evidence is separate from the projection sidecars. The latter
  retain 53/1,800 timeouts and are `PARTIAL_WITH_TIMEOUTS`.
- A gate-created feasible trajectory is not evidence that the raw model
  proposal was feasible.
- MPC and post-gate cost are offline reference metrics; the F1 prompts did not
  instruct the hosted models to optimize the full economic objective.
