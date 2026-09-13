# Paper evidence scope

The canonical paper snapshot in this package is the 17-page, three-author Zenodo manuscript titled **State Sensitivity Is Not Grounding: Dual Replay for LLM Battery Scheduling**.

The paper draws on four distinct evidence layers:

1. **Historical E2 intervention:** canonical versus stale state, byte-identical repeat-null calls, model-stratified action distance, and raw feasibility.
2. **F1/F0 contract comparison:** complete-contract dual replay, violation mechanisms, temporal comparison, and matched deterministic gate baselines.
3. **Qwen3.7-Plus sensitivity tier:** separately reported carrier-consistent and authoritative replay outcomes for the F1 and F0 payload sets.
4. **Provider-free verification:** independent physics replay, deterministic baselines, robustness analyses, feasibility semantics, and claim-level traceability.

E1 is retained because it is used by the supporting completion and activity audit. It is not pooled into the headline F1/F0/Qwen3.7-Plus estimates.

Historical E2 projection sidecars are preserved only as partial evidence. The frozen result is 1,178 computed rows, 569 parser/not-applicable rows, and 53 timeouts from a denominator of 1,800. The package does not convert those timeouts into solved rows and does not use a later projection implementation as a retroactive replacement.

The manuscript PDF is a reference snapshot, not the executable analysis input. All manuscript claims must be checked against `CLAIM_TO_EVIDENCE_MAP.json` and the machine-readable registry in `reference_outputs/audit/`.
