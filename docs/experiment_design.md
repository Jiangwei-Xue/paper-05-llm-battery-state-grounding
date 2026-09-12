# Experimental design

## Research question

The experiments ask whether a hosted language model changes a battery schedule
when the represented state of charge changes, whether that change exceeds
same-payload hosted-call variation, and whether the returned schedule is
physically valid from the state shown to the model and from the authoritative
plant state. Post-gate feasibility is kept separate from raw proposal quality.

## Shared physical task

The task is a 24-hour PV-battery schedule at 15-minute resolution. The frozen
battery has 500 kWh nominal energy capacity, 250 kW charge and discharge limits,
0.95 charge and discharge efficiencies, 75 kWh reserve, a 150 kW export limit,
and a 325 kWh experimental terminal target. Five event families change a
forecast or physical envelope during the horizon. The complete parameters and
the four California locations are in `frozen_inputs/project/configs/frozen_protocol.yaml`.

The task source combines NREL NSRDB GOES CONUS v4.0.0 solar data, NREL OEDI
ComStock AMY2018 release 1 large-office load data, and CAISO OASIS day-ahead LMP
data at `TH_NP15_GEN-APND`. The retained task objects contain the vectors used by
the experiments. The original upstream data remain subject to their source
terms. Complete vectors for every rejected candidate were not retained; this is
an explicit provenance boundary, not an imputed dataset.

## Experiments

### E1: interface and action-activity support

E1 contains 240 logical episodes and 480 stage calls across DeepSeek and
Qwen3.6 Flash conditions. It contrasts an earlier direct JSON interface with a
structured sparse-action contract while keeping the task and physical state
fixed. It supports claims about parse success and nontrivial action generation;
it does not supply raw-feasible schedules.

### E2: historical state-carrier intervention

E2 contains 360 paired blocks. Each block shares Stage 1 output, event,
suffix horizon, model, request settings, and task while Stage 2 uses branches
`C`, `M`, `S`, `K`, and `C2`. `C` is the canonical carrier, `S` is stale,
`M` is model-maintained, `K` is a metadata contrast, and `C2` is a byte-identical
repeat of `C`. The `C-C2` distance estimates hosted-call variation. E2 has 2,160
planned calls, 2,158 actual calls, and 2,146 successful responses. Its 1,800
branch rows are the denominator for raw-action comparisons. Historical
projection evidence is secondary because 53 rows reached the frozen 120-second
timeout.

### F1: complete-contract dual replay

F1 contains 120 four-branch blocks and 480 branch calls. The branch pairs
`C1/C2` and `S1/S2` are duplicate-payload controls. Canonical and stale state
conditions are counterbalanced. The response contract specifies battery
dynamics, limits, local action offsets, and terminal obligation. Every parsed
schedule is replayed twice: from the represented carrier state and from the
authoritative plant state.

### F0: temporally separated earlier-contract comparison

F0 contains 30 blocks and 120 branch calls. It uses the same four-branch
counterbalancing structure with the earlier payload contract. It is analyzed
separately from F1 and is never pooled into the confirmatory F1 denominator.

### QWEN37: model-family sensitivity tier

This tier contains 75 blocks and 300 branch calls using `qwen3.7-plus` over the
frozen F1 and F0 tasks. It preserves the same parser, action schema,
temperature, token cap, duplicate-request controls, and branch-order logic. It
is a separate sensitivity tier and is not pooled with historical Flash-model
results.

## Conditions and controls

- Canonical versus stale state is the intervention.
- Duplicate canonical or duplicate stale requests estimate hosted-call noise.
- Forecasts, prices, event, horizon, constraints, parser, and model settings are
  fixed within a comparison block.
- F1 and F0 branch order is counterbalanced by the frozen run plans.
- Zero, deterministic terminal-chasing, random, and same-information MPC
  proposals are matched offline to the same scenarios and gate contract.
- No AI judge or manual repair is used.

## Model and request settings

The archived conditions are `deepseek-v4-flash`,
`qwen3.6-flash-2026-04-16`, and the separate `qwen3.7-plus` tier. Requests use
temperature 0 and a 16,000-token output cap. Exact prompts and request payloads
are stored in every record, so the saved bytes, rather than a reconstructed
template, are authoritative.

## Parsing, replay, and metrics

E1/E2 use the archived parser in `protocol_code/experiments_v2/p0.py` and F1,
F0, and QWEN37 use `protocol_code/e2b_parser_v2.py`. The verifier re-runs those
parsers against all successful saved responses. Schedules are then evaluated by
an independent deterministic physics checker.

`carrier_consistent_raw_feasible` means replay from the state shown to the
model. `authoritative_raw_feasible` means replay from canonical plant state and
is the deployment-facing safety endpoint. Exact terminal equality and the
1 kWh engineering terminal tolerance are kept separate. Post-gate feasibility
is attributed to the deterministic gate, not to the model proposal.

## Randomness and failures

Scenario selection, event generation, and bootstrap seeds are 20260713,
20260714, and 20260716. Hosted calls may vary despite temperature zero; the
duplicate-payload controls measure that variability. Every attempt and its send
and receive timestamps are retained. Transport failures and structured skips
remain in the planned denominator and are not silently dropped. The offline
reproduction performs no retry because it never calls a provider.

## Execution order

The methodological order was E1, E2, F1, F0, then QWEN37. The one-command
offline workflow rechecks the scientific release contract, re-parses saved
responses, recomputes deterministic analyses, runs the independent checker,
regenerates tables, performs the adversarial audit, and verifies VCR/H5/E5.

See `experiments_manifest.json`, `EXPERIMENT_TIMELINE.md`, and
`METHOD_VERSION_HISTORY.md` for machine-readable denominators and public method
evolution.
