# Raw-action feasibility semantics

The release reports two replay endpoints and never uses an unqualified raw-feasibility field.

- `carrier_consistent_raw_feasible`: replay starts from the state carried in the model-facing branch. It diagnoses the action in the world represented to the model.
- `authoritative_raw_feasible`: replay starts from canonical system state. It is the safety/deployment-facing primary endpoint.

Both endpoints use the engineering terminal tolerance of 1 kWh. Exact-terminal gates are separate optimization diagnostics and are not substituted for either raw-action endpoint.

| Tier | Denominator | Carrier-consistent | Authoritative |
|---|---:|---:|---:|
| historical_F1 | 480 | 0 | 0 |
| historical_F0 | 120 | 0 | 0 |
| qwen37_F1 | 240 | 6 | 3 |
| qwen37_F0 | 60 | 1 | 0 |
