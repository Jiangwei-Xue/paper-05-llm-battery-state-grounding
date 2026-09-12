# Method version history

## Historical E1/E2 contract

E1 tests interface effects. E2 introduces the five carrier branches `C`, `M`,
`S`, `K`, and `C2`; `C2` duplicates `C` byte for byte to measure hosted-call
variability. E2 raw-action claims do not depend on the archived projection
sidecars. The projection sidecars retain a 53/1,800 timeout boundary.

## Complete-contract F1

F1 moves to a strict local-offset action schema and makes the physical contract
explicit. It introduces duplicate pairs on both the canonical and stale sides
and evaluates every schedule under represented-state and authoritative-state
replay. This change is methodological and F1 is therefore not pooled with E2.

## Temporally separated F0

F0 applies the four-branch controlled design to an earlier payload contract. It
is retained as a separate temporal comparison rather than folded into F1.

## Qwen3.7-Plus sensitivity

QWEN37 preserves the frozen F1/F0 tasks, action schema, parser, temperature,
token cap, and counterbalancing while changing the model condition. It is a
separate sensitivity tier.

## Offline standardization

The offline analysis separates `carrier_consistent_raw_feasible` from
`authoritative_raw_feasible`, distinguishes exact terminal equality from the
1 kWh engineering tolerance, and attributes post-gate feasibility to the gate.
Matched zero, heuristic, random, and MPC references use the same task and gate
contract. No historical response is rewritten.

## Public v1.3 derivative

This release adds exact archived parser source and parser replay, public method
documentation, experiment and result manifests, and stronger release-content
audits. Scientific records, model outputs, paper source, and paper PDF remain
unchanged from the verified v1.2 release. A non-executable third-party docstring
example is translated to eliminate unrelated Chinese text, and environment
metadata no longer carries development-history identifiers.
