# F0 formal protocol v1

This is the separately frozen 120-call formal execution of the optional F0
bridge. It is independent from the completed eight-call smoke and 40-call
technical pilot, and it is not part of the E2b/F1 denominator.

## Formal design

The formal plan contains 30 source blocks and 120 provider calls. Each block
uses one of the two frozen model conditions and four ordered branches: C1
(canonical), C2 (byte-identical canonical duplicate), S1 (stale-SOC), and S2
(byte-identical stale duplicate). The 30 source rows, their branch orders,
tasks, and request hashes are frozen before execution.

The plan covers 15 scenarios, both model conditions, and all five event
families. Selection is inherited from the source F0 bridge plan and is
outcome-blind and model-blind. The completed smoke and pilot are acceptance
gates only; their outcomes do not alter the formal task list or execution
parameters.

## Contract and failure accounting

The formal run uses the frozen E2b-v2 local-offset JSON contract, prompt,
schema, parser, deterministic replay, and branch definitions. Raw responses,
parser failures, transport failures, and model-output failures remain in the
120-call denominator. There is no parser repair, clipping, fallback, human
label, LLM judge, selective retry, or post-hoc replacement.

Each record preserves the frozen task reference and hashes, prompt and request
payload, raw provider response and returned-model metadata, parsed action,
carrier, deterministic replay, score, branch identity, and UTC timing fields.

## Authorization boundary

The formal package is `PRE_CALL_FROZEN_HOLD`. Offline verification checks all
plan, prerequisite, hash, and worktree conditions without contacting a
provider. A later explicit user authorization is required before the formal
runner may create a provider run. The formal run uses a new empty output
directory and cannot resume or replace pilot or smoke artifacts.
