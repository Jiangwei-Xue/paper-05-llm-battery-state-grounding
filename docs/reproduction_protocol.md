# Reproduction protocol

## Declared scope

This is a complete provider-free reproduction from frozen task, request, raw
response, and parser evidence through deterministic replay, scoring, branch
comparison, sensitivity analysis, and paper tables. It does not regenerate
paid or nondeterministic hosted-model responses.

## Requirements

- macOS arm64;
- CPython 3.12;
- no network access;
- approximately 1 GB free working space.

The package includes the exact tested NumPy, pandas, SciPy, PyYAML, and
python-dateutil runtime under `runtime/vendor/`. Versions and platform scope are
recorded in `ENVIRONMENT.json` and `environment/requirements-lock.json`.

## One command

From the extracted release root:

```sh
python3.12 RUN_ALL_OFFLINE.py --archive-context <ARCHIVE_PATH>
```

The archive argument is recorded only as a provenance label. The program reads
no file outside the extracted release. Outputs are written to a fresh sibling
directory so the extracted evidence tree remains immutable.

## Stages

1. Validate the scientific release contract, paper files, and manifests.
2. Run parser and release-contract unit tests.
3. Re-run exact archived parsers against every successful saved response.
4. Recompute historical F1/F0 metrics and matched gate baselines.
5. Recompute the separate Qwen3.7-Plus sensitivity tier.
6. Run the independent physics checker over all retained replay objects.
7. Rebuild the completion audit, optimization sensitivities, and explicit
   feasibility-semantics report.
8. Regenerate the release tables.
9. Run adversarial acceptance checks.
10. Verify the VCR/H5/E5 evidence chain.
11. Compare regenerated deterministic outputs with frozen references.

## Expected acceptance values

- 825 retained record/block files;
- 2,940 claim-bearing rows;
- 3,538 saved provider calls represented;
- 3,526 successful responses independently re-parsed;
- 14 failed or skipped calls marked not applicable to parser replay;
- 1,800 dual-replay comparison rows;
- 4,620 independent physics comparisons;
- zero parser mismatch;
- zero physics disagreement;
- zero provider call and zero network attempt.

## Phased commands

The one-command workflow is authoritative. Individual stages can be inspected
by reading the ordered `commands` list in `RUN_ALL_OFFLINE.py`. Each command has
an explicit input and output directory, and each failure terminates the run.

## Hosted experiment reconstruction

`experiment_scaffolding/EXPERIMENT_MAP.md` documents E1, E2, F1, F0, and the
Qwen3.7-Plus sensitivity tier. The complete source layout is retained under
`frozen_inputs/project/`, including task preparation, plan freezing, pre-call
manifests, prompt and schema definitions, counterbalancing, guarded live
runners, parsers, replay, scoring, aggregation, verification, and tests.

These live sources are an optional prospective-replication interface, not part
of the accepted offline result reproduction. They require network access,
provider credentials supplied through environment variables, and explicit
execution authorization. Their outputs must be stored as new evidence and
must not replace the frozen records in this release.

## Result equivalence

Deterministic scientific values must match their frozen references exactly.
Some receipt files contain elapsed-time fields; those are compared after
excluding timing-only provenance. No tolerance is relaxed to obtain a pass.

## Failure behavior

Any missing input, malformed record, hash mismatch, parser mismatch, unexpected
network attempt, physical disagreement, table mismatch, or forbidden residue
causes a nonzero exit. A partial run is not a successful reproduction.

## Upstream-data boundary

The package reproduces results from retained frozen task vectors. It does not
redownload source datasets. The public source descriptions are sufficient to
identify the upstream releases, but complete vectors for rejected candidate
scenarios were not retained; the package therefore does not claim to recreate
the entire preselection candidate pool.
