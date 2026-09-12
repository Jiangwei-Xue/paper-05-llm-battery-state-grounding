# Experiment map

## E1: interface contract

- Purpose: test whether structured and sparse interfaces change parsing and
  raw-action behavior.
- Unit: 240 logical episodes and 480 hosted stage calls.
- Conditions: I0 and I3 under DeepSeek and Qwen Flash.
- Controls: frozen physical task, model settings, temperature, token cap,
  parser, retry policy, and five-call concurrency.
- Entry sequence: prepare tasks, freeze run plan and hashes, preflight, run,
  deterministic projection/replay, aggregate, verify.
- Paper role: interface changes can change action generation while raw physical
  feasibility remains a separate endpoint.

## E2: state-carrier intervention

- Purpose: compare canonical, model-maintained, stale, metadata-only, and
  byte-identical repeated canonical branches.
- Unit: 360 paired blocks, 2,160 planned calls, and 2,158 actual calls.
- Branches: C, M, S, K, C2 with shared Stage 1 evidence.
- Controls: task, event, forecast, prices, horizon, physical limits, interface,
  sampling, and branch schedule.
- Entry sequence: prepare and admit tasks, freeze paired plan, preflight, run,
  parse, raw-action comparison, deterministic replay, projection audit, verify.
- Paper role: carrier response is evaluated relative to repeat-call variation;
  historical projection coverage is reported separately.

## F1: complete-contract dual replay

- Purpose: test state sensitivity after explicitly specifying battery dynamics,
  time indexing, limits, terminal requirements, and action representation.
- Unit: 120 four-branch blocks and 480 branches.
- Branches: C1, C2, S1, S2 with counterbalanced order.
- Controls: complete prompt contract, schema, parser, task, event, horizon,
  model condition, retry limit, and frozen branch schedule.
- Entry sequence: prepare controls and prompt fixtures, freeze task and run
  plans, preflight, guarded live runner, parse, carrier-consistent and
  authoritative replay, matched gate analysis, verify.
- Paper role: distinguishes response to represented SOC from grounding in the
  authoritative plant state.

## F0: old-contract temporal comparison

- Purpose: run a temporally separated subset under the earlier payload contract
  without pooling it with F1.
- Unit: 30 four-branch blocks and 120 branches.
- Controls: frozen 30-block plan, the same branch structure, explicit pre-call
  manifest, independent result directory, and no selective retry.
- Entry sequence: prepare formal plan, preflight, guarded live runner, parse,
  analyze, finalize, VCR verify.
- Paper role: historical-contract comparison only.

## Qwen3.7-Plus sensitivity tier

- Purpose: evaluate whether the F1/F0 findings persist for a later direct Qwen
  model condition.
- Unit: 75 four-branch blocks and 300 branches: 60 F1 blocks and 15 F0 blocks.
- Controls: direct route, fixed model identifier, temperature zero, disabled
  fallback/tools/web search/cache, counterbalanced branches, frozen tasks, and
  independent analysis.
- Entry sequence: prepare plan and model spec, freeze manifest, smoke preflight,
  guarded live runner, manifest build, analyze, verify.
- Paper role: separate sensitivity tier; it is never pooled with the historical
  Flash conditions.

## Shared baselines and evaluation

The saved-evidence analysis includes byte-identical repeat controls, zero
proposal, deterministic terminal-chasing proposal, random proposal,
same-information MPC, represented-state replay, authoritative-state replay,
gate correction distance, violation severity, physical cost, and independent
equation checks. `CLAIM_TO_EVIDENCE_MAP.json` and
`docs/paper_results_map.md` bind these outputs to the paper.

