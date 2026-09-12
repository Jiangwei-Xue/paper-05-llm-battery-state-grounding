# Result provenance audit

- Paper/result items mapped: 9/9 paper tables and 10 claim-registry entries.
- Unmapped core results: 0.
- Raw outputs retained: yes, for all 825 record/block files.
- Intermediate outputs retained: saved parsed actions, deterministic scores,
  replay objects, branch identities, run plans, admissions, selection traces,
  and projection-timeout audit.
- Aggregation reproducible: yes, through the one-command workflow.
- Figure/table generation reproducible: all paper tables have row-level source
  mappings; release tables are regenerated and checked. No paper figure
  environment exists.
- Parser provenance: 3,526 successful raw responses are re-parsed with exact
  archived parser source; 14 failed/skipped calls remain explicit.
- Independent evaluation: 4,620 physical verdicts are checked by a separate
  implementation.
- Known partial evidence: archived E2 projection has 53/1,800 frozen timeouts
  and is not promoted to complete-denominator evidence.

Status: **PASS WITH THE E2 PROJECTION AND SOURCE-SELECTION BOUNDARIES EXPLICIT**.
