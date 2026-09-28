# Mirrored-encoding robustness check

Generated 2026-09-28 11:10. Read-only over results/xgb_mh/{mirror,full,full_ref}.

## 1. Pre-registration

- prediction.md sha256 matches prediction.sha256: **True** (`c96ce35686126eb7...`)
- mirror run created (UTC) 2026-09-28T06:51:57.927410+00:00; recorded config: {'datasets': ['BBBP', 'FreeSolv'], 'experiments': ['B'], 'optimizers': ['random', 'tpe', 'gwo_ref', 'woa_ref', 'pso'], 'seeds': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], 'budget': 100, 'n_jobs': 10, 'encoding': 'mirrored'}

## 2. Audit of the mirror run

- **PASS**: 300 runs (expected 300, missing 0, unexpected 0); every run 100 ordered calls and `encoding: mirrored`: True; calls 1-10 decode to decode(1 - shared design) in 3000/3000 calls (config + feature mask, exact); calls 1-10 fitness identical across the 5 optimizers in 60/60 (dataset, seed) groups; errors.log absent

## 3. Pre-specified tests

Paired Wilcoxon signed-rank (two-sided, zero_method = pratt), pairs = 30 tuning seeds, normal vs mirrored; Holm over the 5 optimizers within each dataset x outcome. diff = mirrored - normal.

**BBBP - selected-feature fraction**

| optimizer | median normal | median mirrored | median paired diff | seeds mirrored > normal | p raw | p Holm |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0.514 | 0.505 | +0.002 | 15/30 | 0.975 | 1.000 |
| tpe | 0.490 | 0.502 | -0.005 | 12/30 | 0.491 | 1.000 |
| gwo_ref | 0.440 | 0.552 | +0.102 | 29/30 | <0.001 | <0.001 * |
| woa_ref | 0.500 | 0.464 | -0.074 | 11/30 | 0.080 | 0.322 |
| pso | 0.500 | 0.495 | -0.014 | 10/30 | 0.621 | 1.000 |

**BBBP - test roc_auc**

| optimizer | median normal | median mirrored | median paired diff | seeds mirrored > normal | p raw | p Holm |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0.904 | 0.904 | +0.000 | 15/30 | 0.853 | 1.000 |
| tpe | 0.911 | 0.909 | +0.003 | 18/30 | 0.551 | 1.000 |
| gwo_ref | 0.907 | 0.908 | -0.003 | 13/30 | 0.839 | 1.000 |
| woa_ref | 0.904 | 0.903 | +0.006 | 18/30 | 0.360 | 1.000 |
| pso | 0.909 | 0.910 | +0.001 | 16/30 | 1.000 | 1.000 |

**FreeSolv - selected-feature fraction**

| optimizer | median normal | median mirrored | median paired diff | seeds mirrored > normal | p raw | p Holm |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0.515 | 0.503 | +0.003 | 15/30 | 0.459 | 1.000 |
| tpe | 0.485 | 0.500 | +0.018 | 18/30 | 0.133 | 0.532 |
| gwo_ref | 0.430 | 0.553 | +0.135 | 28/30 | <0.001 | <0.001 * |
| woa_ref | 0.579 | 0.500 | -0.064 | 13/30 | 0.453 | 1.000 |
| pso | 0.488 | 0.494 | -0.003 | 12/30 | 0.636 | 1.000 |

**FreeSolv - test rmse**

| optimizer | median normal | median mirrored | median paired diff | seeds mirrored > normal | p raw | p Holm |
| --- | --- | --- | --- | --- | --- | --- |
| random | 2.456 | 2.434 | +0.039 | 17/30 | 0.598 | 1.000 |
| tpe | 2.464 | 2.464 | -0.018 | 14/30 | 0.871 | 1.000 |
| gwo_ref | 2.425 | 2.374 | -0.078 | 13/30 | 0.428 | 1.000 |
| woa_ref | 2.544 | 2.460 | -0.041 | 11/30 | 0.516 | 1.000 |
| pso | 2.390 | 2.457 | +0.094 | 18/30 | 0.245 | 1.000 |

## Verdict (decision rule from prediction.md)

gwo_ref supports the prediction if, on EACH dataset, its mirrored fraction is higher (Holm p < 0.05) AND its median mirrored fraction is > 0.5. random (control) supports it if its fraction shows no Holm-significant difference on either dataset. (The rule does not define 'partly'; used here: some but not all of the 4 gwo_ref conditions met.)

- gwo_ref BBBP: mirrored fraction higher with Holm p < 0.05: **True** (median diff +0.102, p Holm <0.001); median mirrored fraction > 0.5: **True** (0.552)
- gwo_ref FreeSolv: mirrored fraction higher with Holm p < 0.05: **True** (median diff +0.135, p Holm <0.001); median mirrored fraction > 0.5: **True** (0.553)

**gwo_ref: SUPPORTED** (4/4 conditions met).
**random (control): SUPPORTED (no change)** (p Holm: BBBP 1.000, FreeSolv 1.000).

## 4. EXPLORATORY (not pre-specified; no inference)

| dataset | optimizer | fraction IQR normal | fraction IQR mirrored | median val raw normal | median val raw mirrored | median unique normal | median unique mirrored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BBBP | random | 0.48-0.53 | 0.49-0.52 | 0.0521 | 0.0534 | 100 | 100 |
| BBBP | tpe | 0.48-0.53 | 0.48-0.52 | 0.0482 | 0.0467 | 100 | 100 |
| BBBP | gwo_ref | 0.42-0.47 | 0.53-0.58 | 0.0467 | 0.0464 | 100 | 100 |
| BBBP | woa_ref | 0.42-0.60 | 0.39-0.52 | 0.0504 | 0.0533 | 97 | 98 |
| BBBP | pso | 0.48-0.51 | 0.47-0.51 | 0.0504 | 0.0491 | 100 | 100 |
| FreeSolv | random | 0.48-0.54 | 0.49-0.52 | 2.4145 | 2.4449 | 100 | 100 |
| FreeSolv | tpe | 0.46-0.50 | 0.47-0.53 | 2.1351 | 2.0977 | 100 | 100 |
| FreeSolv | gwo_ref | 0.40-0.46 | 0.52-0.57 | 2.0444 | 2.1089 | 100 | 100 |
| FreeSolv | woa_ref | 0.38-0.71 | 0.35-0.58 | 2.4115 | 2.3562 | 97 | 97 |
| FreeSolv | pso | 0.47-0.53 | 0.46-0.52 | 2.2771 | 2.1883 | 100 | 100 |

val raw = best validation raw metric (RMSE for FreeSolv, 1 - ROC-AUC for BBBP; lower is better).
