# Experimental design audit

- Experiments identified: 5 (`E1`, `E2`, `F1`, `F0`, `QWEN37`).
- Baselines identified: same-payload repeat nulls; zero proposal; deterministic
  terminal-chasing proposal; random proposal; same-information MPC.
- Controls identified: task, event, horizon, forecast, price, constraints,
  parser, temperature, token cap, branch order, and within-block model condition.
- Ablations identified: interface contract, carrier state, payload-contract tier,
  model-family sensitivity, replay state, gate baseline, objective-order
  sensitivity, event family, activity, and state-gap dose.
- Experiment entry points: `RUN_ALL_OFFLINE.py` for provider-free reproduction;
  experiment-specific prepare, freeze, preflight, live runner, analysis, and
  verifier paths are indexed in `experiment_scaffolding/RUNNER_INDEX.json`.
- Experiment configs: physical protocol, E1/E2 condition configs, provider
  model settings, F1/F0/Qwen3.7 protocols, frozen task manifests, run plans,
  variable freezes, and pre-call manifests are retained under
  `frozen_inputs/project/`.
- Model/prompt configs: exact prompts and request payloads are embedded in every
  retained record. Public prompt templates, response schemas, provider model
  settings, branch schedules, retry limits, and counterbalancing code are also
  retained in the project snapshot.
- Evaluation definitions: independent deterministic replay, represented-state
  and authoritative-state feasibility, violation severity, action distance,
  gate correction, cost, and matched-baseline comparisons.
- Aggregation logic: packaged under `scripts/`; outputs are regenerated under
  primary, QWEN37, physics, audit, sensitivities, semantics, and tables.
- Missing design information: complete source vectors for rejected candidates
  were not retained, so alternative preselection counterfactuals stop at the
  documented provenance boundary. Hosted rolling services may return different
  responses in a prospective rerun even when the saved request is reproduced.

Status: **PASS WITH DECLARED SOURCE-SELECTION BOUNDARY**.
