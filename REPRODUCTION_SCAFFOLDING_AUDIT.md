# Reproduction scaffolding audit

- Setup: bundled CPython 3.12-compatible dependencies and environment lock.
- Data preparation: frozen task objects, admissions, selection trace, and run
  plans; no network download is required.
- Experiment replay: `RUN_ALL_OFFLINE.py` for the verified offline path. The
  original experiment layout is reconstructed under `frozen_inputs/project/`
  with public, fail-closed live runner sources and explicit preflight paths.
- Parser replay: `scripts/reparse_saved_responses_v1_3.py` with exact archived
  parser source under `protocol_code/`.
- Evaluation harness: primary recomputation, QWEN37 standardization,
  independent physics checker, completion audit, feasibility semantics, and
  optimization sensitivities under `scripts/`.
- Aggregation: `recompute_offline_v1_7.py` and associated audit scripts.
- Figure/table generation: nine paper tables are mapped; executable release
  table regeneration is `regenerate_release_tables_v1_7.py`. The manuscript has
  no figure environments and no presentation source is included.
- Reference outputs: `reference_outputs/`.
- Validation: unit tests, scientific-release audit, adversarial acceptance,
  VCR/H5/E5, independent replay, and frozen-output comparison.
- Live experiment sources: E1/E2 prepare-freeze-run-project-analysis chains;
  F1 E2b-v2 protocol, prompt, schema, counterbalanced runner and parser; F0
  formal runner and VCR; Qwen3.7-Plus preparation, model specification, runner,
  manifest builder, analysis and verification.
- Missing scaffolding: none for the declared saved-evidence reproduction.
  Hosted-provider regeneration remains outside the public command because it is
  paid and nondeterministic; credentials are not included. Exact requests and
  saved responses are retained for VCR replay.

Status: **PASS FOR THE DECLARED OFFLINE/VCR REPRODUCTION MODE**.
