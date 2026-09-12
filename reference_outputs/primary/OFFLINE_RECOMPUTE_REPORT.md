# Offline recomputation report

Generated: 2026-08-29T03:50:24.525737+00:00

All analyses use saved records only. Provider calls: **0**. Network attempts: **0**.

## 1. Recomputed F1 carrier metrics

### deepseek_formal

- ED: mean 1173.33 kWh, median 0.00, 95% benchmark-resampling interval [324.37, 2244.73], support +/0/- = 8/52/0.
- Horizon-normalized ED: mean 51.56 kW; median 0.00 kW.
- SOC-gap-normalized ED: mean 17.719; median 0.000.
- Mean after removing the largest 1/2/5 ED values: 803.39, 618.97, 91.36 kWh.
- 10% and 20% trimmed means: 0.52 and 0.00 kWh.

### qwen_flash

- ED: mean 345.45 kWh, median 0.00, 95% benchmark-resampling interval [-103.50, 1112.25], support +/0/- = 3/56/1.
- Horizon-normalized ED: mean 16.65 kW; median 0.00 kW.
- SOC-gap-normalized ED: mean 3.078; median 0.000.
- Mean after removing the largest 1/2/5 ED values: -0.39, -44.02, -56.45 kWh.
- 10% and 20% trimmed means: 0.00 and 0.00 kWh.

## 2. Random-pairing sensitivity

- deepseek_formal: observed mean ED 1173.33 kWh; random-pairing null 95% interval [-586.67, 766.46]; upper-tail p=0.00018.
- qwen_flash: observed mean ED 345.45 kWh; random-pairing null 95% interval [-250.35, 486.90]; upper-tail p=0.13536.

## 3. Independent replay verification

- Saved branch replays checked: 1200.
- Maximum trace absolute difference: 0.000e+00.
- Maximum scalar absolute difference: 0.000e+00.
- Rows with any Boolean/count mismatch: 0.
- Hand-constructed tests passed: 7/7.

## 4. F1 gate and baseline recomputation

- F1 branch rows: 480; baseline rows: 300.
- Battery-only F1 gates solved: 480/480.
- Expanded-system F1 gates solved: 480/480.
- Independent same-space optimum versus archived MPC objective: maximum absolute difference 0.000000 USD.

### deepseek_formal

- Minimum engineering-feasible same-space correction: median 130.53 kWh; mean 442.18 kWh.
- Exact-terminal battery-only gate correction: median 131.58 kWh; mean 443.16 kWh.
- Expanded-system gate correction: median 131.58 kWh; mean 443.21 kWh.
- Battery-only post-gate regret: median 3.12 USD.
- Expanded-system curtailment: median 0.00 kWh.

### qwen_flash

- Minimum engineering-feasible same-space correction: median 5220.84 kWh; mean 4738.49 kWh.
- Exact-terminal battery-only gate correction: median 5221.84 kWh; mean 4739.52 kWh.
- Expanded-system gate correction: median 5221.84 kWh; mean 4739.51 kWh.
- Battery-only post-gate regret: median 3.12 USD.
- Expanded-system curtailment: median 0.00 kWh.

## 5. Same-payload temporal rerun

- deepseek_formal: exact branch repeats 59/60; exact four-branch blocks 14/15; request-payload hashes 60/60; system fingerprints equal 0/60.
- qwen_flash: exact branch repeats 49/60; exact four-branch blocks 9/15; request-payload hashes 60/60; system fingerprints equal 0/60.

## Interpretation boundary

The recomputation characterizes the frozen challenge set and the two evaluated hosted model conditions. Bootstrap intervals are resampling-stability intervals for these benchmark scenarios. The offline gate results measure deterministic repair and economic outcomes under the disclosed optimization formulations; they do not add new hosted-model evidence.
