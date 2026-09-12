# F0 formal raw-action analysis

Status: **PASS**

This report is recomputed from the 30 saved formal records and does not pool F0 with E1, E2, E2b, or F1.

- Blocks: **30**
- Branch rows: **120**
- Raw-feasibility definition: **carrier-consistent deterministic replay**
- Provider calls during analysis: **0**
- Network attempts during analysis: **0**

## Interpretation boundary

The carrier contrast is reported with identical-payload nulls retained. A positive null-adjusted distance is descriptive evidence within this frozen challenge set; it is not a population estimate or a model ranking. Projection, fallback, and post-gate claims are outside this report.

## Headline counts

- Raw-feasible branch rows: **{'C1': 0, 'C2': 0, 'S1': 0, 'S2': 0}**
- Nontrivial branch rows: **{'C1': 16, 'C2': 16, 'S1': 16, 'S2': 16}**
- C1–S1 distance: **{'n': 30, 'mean': 528.8833333333333, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'min': 0.0, 'max': 5000.0}**
- C1–C2 null distance: **{'n': 30, 'mean': 344.00875, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'min': 0.0, 'max': 5000.0}**
- Null-adjusted C1–S1 distance: **{'n': 30, 'mean': 274.36229166666664, 'median': 0.0, 'p25': 0.0, 'p75': 0.0, 'min': -1312.5, 'max': 5000.0}**

Machine-readable details are in `F0_FORMAL_ANALYSIS.json` and `F0_FORMAL_ROW_LEVEL_ANALYSIS.jsonl`.
