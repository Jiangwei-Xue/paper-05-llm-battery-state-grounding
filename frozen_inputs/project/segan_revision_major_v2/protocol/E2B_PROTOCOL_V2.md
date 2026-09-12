# E2b-v2 Stage-2-Only Carrier-Sensitivity Protocol

E2b-v2 replaces the failed E2b-v1 execution with a new prospective protocol. The archived v1 run stopped before every treatment call and remains immutable. Its records do not enter E2b-v2 denominators.

## Design

The confirmatory F1 tier uses the 60 frozen E2 scenarios, two frozen direct-provider model conditions, and four Stage-2 calls per scenario-model block. C1 and C2 receive identical canonical-SOC requests. S1 and S2 receive identical stale-SOC requests. The two calls per carrier estimate ordinary hosted-call variation. No live Stage 1 is generated or shown.

The primary within-block statistic is the label-invariant energy-distance contrast

`ED = 2B - Wc - Ws`,

where `B` averages all four cross-carrier distances, `Wc=d(C1,C2)`, and `Ws=d(S1,S2)`. Action responsiveness is estimated only for blocks in which all four branches parse. Interface validity retains the full denominator. A model requires at least 80% complete blocks before its responsiveness result receives confirmatory interpretation.

## Action and physics contract

The response contains only local-offset action segments. The parser rejects aliases, timestamps, global indices, extra fields, overlaps, non-finite power, and more than H segments. Uncovered offsets expand to zero.

Raw scoring propagates requested power without clipping. Positive power charges; negative power discharges. The scorer records power, SOC, export, and terminal violations at their original magnitudes. Export is a strict raw-action constraint: no automatic PV curtailment repairs a proposal before raw scoring. A separate executable replay may report downstream modification, but it cannot replace the raw endpoint.

Both carrier-consistent and authoritative replay are retained. Canonical branches begin from canonical event SOC in both. Stale branches begin from stale SOC for carrier-consistent replay and from canonical SOC for authoritative replay.

## Execution gates

Offline controls, rendered-prompt inspection, parser tests, request-identity checks, treatment-scope checks, analysis tests, artifact hashes, and a clean worktree must pass before live execution. A separately authorized 16-call smoke precedes the formal denominator. The smoke and formal run use new directories and manifests. Formal F1 contains 480 calls. The optional 120-call F0 bridge is a separate bundled contract comparison and cannot isolate any single repair.

A run is completed only when every planned branch has a terminal attempt record, provider-call counts match the plan, duplicate requests are byte-identical, returned models match, no partial files remain, and all frozen hashes verify. Parser and physical failures remain outcomes; they are never retried or rewritten.
