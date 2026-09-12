# Offline completion audit v1

This revision derives additional audit material from frozen saved evidence. It made no provider or network calls and did not modify the manuscript or archived evidence.

## Completed evidence

- Independent equation checker: 4620 checks; 0 discrepancies; 12/12 boundary fixtures passed.
- Gate mechanism rows: 1560 across battery-only and expanded-system projections.
- Selection provenance: legacy 480 candidates, E1 320, E2 280.
- Difficulty/slack diagnostic: 60 selected E2 scenarios. This is post hoc and does not replace registered difficulty labels.
- Activity sensitivity: 6 experiment/model/interface strata at 1, 5, and 10 kWh.
- E1 positive-control manifest: 240 episodes; all three interface changes remain bundled.
- Claim registry: 14 scoped claims.

## Deliberate limits

- No new hosted-model call was made.
- The 30-second timeout run uses one representative per mathematical fingerprint and four fixed workers. It is a concurrency-aware sensitivity result, not a repair or replacement for the formal 120-second run.
- Rejected-candidate admission rows do not all retain full PV/load/price vectors. The selection file therefore reports the retained variables and marks this provenance boundary.
- Objective-order and deterministic-reference sensitivity require new optimization output. They are kept separate from the completed evidence and must not be inferred from saved trajectories.
- Qwen3.7 remains a separate sensitivity tier.

## Status

The five experimental hard requirements identified for submission are present: independent physics checking, matched F1 gates and baselines, economic/regret metrics, violation severity, and SOC dose/direction analysis. Remaining items concern sensitivity breadth and release hygiene rather than missing hosted observations.
