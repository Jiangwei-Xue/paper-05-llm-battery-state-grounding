# VCR, H5, and E5 definitions

## VCR

VCR means saved-response replay. The verifier checks every retained record
against the private-to-public hash receipt, recomputes task, prompt, request,
and parsed-object hashes, and independently re-runs the exact frozen parser on
all successful raw response strings. It does not contact a model provider.

## H5

H5 is evidence-chain completeness at the claim-bearing row level. A valid row
must retain the task identity, prompt and request bytes, raw response or
explicit failure/skip, parsed action, deterministic score or replay object,
comparison identity, timestamps, and hashes. Denominators and structured
failures are checked rather than inferred from summary files.

## E5

E5 is deterministic evaluation closure. From the H5/VCR records, the package
rebuilds replay, feasibility and violation metrics, matched gate baselines,
sensitivity analyses, claim comparisons, and paper tables. An independent
physics implementation must agree with every retained verdict. E5 does not mean
that paid hosted-model outputs were regenerated.

The frozen acceptance contract is `VCR_H5E5_EXPECTED.json`; the regenerated
receipt is `reproduced_outputs/vcr_h5e5/VCR_H5E5_VERIFICATION.json`.
