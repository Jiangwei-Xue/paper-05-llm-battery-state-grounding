# Scientific release audit

## A. Scope

The release covers the evidence used by the bundled 17-page paper: E1, E2, F1, F0, the separate Qwen3.7-Plus tier, archived parsers, deterministic dual replay, scoring, matched baselines, sensitivity analyses, and paper-table provenance.

## B. Experiment inventory and design

Five experiment tiers are identified with fixed denominators, branches, model scope, run plans, task manifests, parser contracts, failure policy, and comparison rules. Duplicate-request null controls and zero, deterministic heuristic, random, stale-state, and same-information MPC references are retained where applicable.

## C. Reproduction scaffolding

The package includes a one-command offline runner, pinned runtime libraries, unit tests, exact parser replay, independent physics checking, result aggregation, sensitivity analysis, table regeneration, and a VCR/H5/E5 verifier. Provider access is neither required nor performed.

## D. Data provenance and claim mapping

`manifests/PUBLIC_RECORD_MANIFEST.jsonl` records the public hash of every saved record. `experiments_manifest.json`, `results_manifest.json`, and `CLAIM_TO_EVIDENCE_MAP.json` connect experiment populations and reported claims to row-level inputs and generated results. All nine paper tables have named source files and reproduction stages.

## E. Results and numerical verification

- Saved record files: 825.
- Claim-bearing evidence rows: 2,940.
- Represented provider calls: 3,538.
- Successful saved responses independently re-parsed: 3,526.
- Parser-replay mismatches: 0.
- Dual-replay comparison rows: 1,800.
- Independent physics verdicts: 4,620.
- Independent physics discrepancies: 0.

The full offline run regenerates the primary analysis, Qwen3.7-Plus standardization, feasibility semantics, gate and baseline comparisons, optimization sensitivities, acceptance checks, and paper tables. Frozen reference outputs are used to detect scientific drift.

## F. Environment and tests

The tested execution target is CPython 3.12 on macOS arm64 using the bundled pinned libraries. Unit, integration, parser, replay, physics, table, and evidence-chain checks run inside the one-command workflow. All experiment timestamps are explicit UTC values; analysis does not use the host time zone.

## G. Paper mapping

The paper title, author order, LaTeX source, bibliography, frozen BBL, and PDF are fixed by `LATEST_PAPER.json`. `docs/paper_results_map.md` maps all paper tables to generated evidence. Editable presentation files and alternative manuscript versions are outside this release scope.

## H. Limitations

Hosted-model generation is not repeated. Historical E2 projection coverage is incomplete because 53/1,800 rows timed out under the frozen watchdog. The raw-action and exact-terminal endpoints remain distinct. Complete input vectors are unavailable for every rejected candidate, limiting selection-rule counterfactuals. Compatibility beyond the tested platform requires the versions in the environment lock and is not asserted here.

## I. Release decision

Scientific release readiness is established only when the final archive passes deterministic duplicate build, clean extraction, documented command execution, one-command offline reproduction, VCR/H5/E5 validation, manifest closure, and final archive verification.
