# State Grounding in Battery Scheduling: reproducibility package

This package contains the latest paper and the smallest complete evidence closure needed to reproduce its reported experimental results from saved provider records. It includes frozen tasks, prompts and request payloads, raw responses, parsed actions, deterministic replay, scores, branch comparisons, public hashes, analysis code, expected outputs, and a pinned offline runtime. It does not repeat hosted-model calls.

The bundled paper is **State Sensitivity Is Not Grounding: Dual Replay for LLM Battery Scheduling** by Jiangwei Xue, Zhida Qin, and Yuda Bi (DOI: [10.5281/zenodo.22707487](https://doi.org/10.5281/zenodo.22707487)). The exact LaTeX, bibliography, frozen BBL, and 17-page PDF are under `paper_reference/`; their hashes are recorded in `LATEST_PAPER.json`.

## Public manuscript

The publicly available manuscript version associated with this reproduction
package is available on Zenodo:

https://doi.org/10.5281/zenodo.22707487

Concept DOI: https://doi.org/10.5281/zenodo.22707486

The author-owned manuscript material in that version is licensed under
CC BY 4.0.

The corresponding repository files and their hashes are identified in
`paper_reference/LICENSE.md` and `LATEST_PAPER.json`.

Reproducibility package version: 1.5.1.

## One-command offline reproduction

The tested execution target is CPython 3.12 on macOS arm64. From the extracted package root run:

<!-- RELEASE_COMMAND:reproduce -->
```sh
python3.12 RUN_ALL_OFFLINE.py --archive-context <ARCHIVE_PATH>
```
<!-- END_RELEASE_COMMAND:reproduce -->

Replace `<ARCHIVE_PATH>` with the final release archive path. The argument is only a release-context label; all computation reads the extracted package. The command writes a sibling output directory, blocks network access, re-parses saved responses, recreates deterministic replay and scores, rebuilds comparisons, sensitivities and tables, and checks VCR/H5/E5 closure. Success ends with:

```text
OFFLINE_REPRODUCTION_PASS output=<sibling-output-directory> provider_calls=0 network_attempts=0
```

## Archive verification

<!-- RELEASE_COMMAND:verify -->
```sh
env HOME=. python3 tools/VERIFY_ARCHIVE.py verify --archive <ARCHIVE_PATH>
```
<!-- END_RELEASE_COMMAND:verify -->

Every mandatory gate must report `PASS`; an unperformed check is not a pass.

## Evidence inventory

| Tier | Saved records | Scientific role |
|---|---:|---|
| E1 | 240 | interface and raw-action audit |
| E2 | 360 | state-carrier intervention and repeat-null control |
| F1 | 120 blocks / 480 branches | complete-contract dual replay and matched gate analysis |
| F0 | 30 blocks / 120 branches | temporally separated old-contract comparison |
| Qwen3.7-Plus | 75 blocks / 300 branches | separate F1/F0 sensitivity tier |

Together these comprise 825 saved record files, 2,940 claim-bearing rows, and 3,538 represented provider calls. The parser replay covers 3,526 successful retained responses with zero parsed-object mismatch. VCR validates the saved request/response chain, H5 checks row-level timestamps and hashes, and E5 checks 1,800 dual-replay comparisons and 4,620 independent physics verdicts.

Start with `docs/experiment_design.md`, `docs/reproduction_protocol.md`, and `docs/paper_results_map.md`. Machine-readable experiment and result inventories are `experiments_manifest.json`, `results_manifest.json`, and `CLAIM_TO_EVIDENCE_MAP.json`.

For the complete hosted-experiment design, start with
`experiment_scaffolding/EXPERIMENT_MAP.md`. The public project snapshot under
`frozen_inputs/project/` includes the protocols, prompt templates, schemas,
model settings, task and run plans, variable freezes, pre-call manifests,
counterbalancing logic, live runner sources, parsers, deterministic physics,
evaluation, aggregation, and tests for E1, E2, F1, F0, and Qwen3.7-Plus. Live
runners are never called by the offline reproduction workflow and require
separately supplied environment credentials and explicit authorization.

## Interpretation boundaries

- `authoritative_raw_feasible` is the safety-facing endpoint; `carrier_consistent_raw_feasible` is a diagnostic replay from the represented state.
- F1, F0, and Qwen3.7-Plus are separate tiers and are not pooled.
- Historical E2 projection evidence is partial: 53/1,800 projection attempts reached the frozen 120-second timeout.
- The raw-action engineering endpoint uses a 1 kWh terminal tolerance; exact-terminal gate analyses are separate.
- Complete input vectors were not retained for every rejected candidate scenario, so selection-rule counterfactuals stop at the retained-data boundary.
- Saved provider outputs are replayed offline; the package does not claim to reproduce nondeterministic hosted-model generation.
- Prospective live runs create new evidence and do not replace the frozen
  records or reference outputs used by the paper.

## Environment

Dependency versions and the tested compatibility boundary are recorded in `PUBLIC_ENVIRONMENT.json` and `environment/requirements-lock.json`. The workflow uses explicit UTC experiment timestamps and does not depend on the host time zone.

## Licensing

This repository uses layered licensing:

- project-authored analysis, replay, verification, testing, and utility
  code: MIT License;
- the author-owned paper, original project documentation, and
  author-generated derived results: CC BY 4.0;
- bundled Python libraries: their included upstream licenses;
- public data, hosted-model records, provider metadata, and other
  third-party material: their original applicable terms.

The manuscript-specific notice is in
`paper_reference/LICENSE.md`.

See `LICENSE`, `LICENSES/README.md`, and
`LICENSES/THIRD_PARTY.md` for the controlling path and material
boundaries. No single license applies to the repository as a whole.
