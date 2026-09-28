# Final statistics - xgb_mh

Generated 2026-09-28 11:16. Analysis code sha256 `264415706d217c2910d722afafbc4f3eb8b5c7c6832cd2da4b507323bac15cba` (files in final_stats.json). Inputs: full `4669997af4a6`, full_ref `f056a88816b3`, mirror `fc7d77de4416`.
GWO / WOA = reference-faithful gwo_ref / woa_ref. n = 30 tuning seeds per method. Bootstrap: 10,000 percentile resamples of seeds, seed 20260927. Wilcoxon: zero_method = pratt. Improvement over default is oriented so > 0 = better.

## 1. Tuning vs default (primary metric)

Cell: median improvement over default [bootstrap 95% CI], % of seeds beating default; * = one-sample Wilcoxon vs the default value, Holm over the 5 optimizers, p < 0.05.

| dataset | exp | default | Random | TPE | GWO | WOA | PSO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESOL | A | 0.889 | -0.035 [-0.042, -0.025] 3% * | -0.030 [-0.039, -0.016] 7% * | -0.027 [-0.040, -0.016] 10% * | -0.022 [-0.034, -0.007] 17% * | -0.027 [-0.036, -0.016] 7% * |
| ESOL | B | 0.889 | -0.032 [-0.052, -0.010] 23% * | -0.012 [-0.022, -0.004] 30% * | -0.030 [-0.046, -0.015] 27% * | -0.029 [-0.046, -0.009] 27% * | -0.016 [-0.024, -0.003] 30% * |
| FreeSolv | A | 2.811 | +0.200 [+0.082, +0.351] 83% * | +0.296 [+0.254, +0.331] 90% * | +0.355 [+0.295, +0.443] 97% * | +0.208 [+0.168, +0.278] 93% * | +0.321 [+0.246, +0.386] 100% * |
| FreeSolv | B | 2.811 | +0.355 [+0.242, +0.445] 93% * | +0.347 [+0.228, +0.433] 93% * | +0.385 [+0.288, +0.527] 90% * | +0.267 [+0.200, +0.406] 93% * | +0.421 [+0.306, +0.471] 97% * |
| Lipophilicity | A | 0.760 | +0.031 [+0.026, +0.034] 93% * | +0.035 [+0.030, +0.040] 100% * | +0.034 [+0.032, +0.036] 100% * | +0.033 [+0.029, +0.037] 90% * | +0.032 [+0.030, +0.038] 90% * |
| Lipophilicity | B | 0.760 | +0.022 [+0.018, +0.025] 93% * | +0.022 [+0.015, +0.024] 93% * | +0.026 [+0.018, +0.031] 93% * | +0.021 [+0.011, +0.030] 93% * | +0.026 [+0.018, +0.029] 93% * |
| BBBP | A | 0.915 | -0.012 [-0.019, -0.009] 3% * | -0.015 [-0.020, -0.009] 17% * | -0.006 [-0.009, -0.001] 27% * | -0.012 [-0.020, -0.007] 17% * | -0.014 [-0.017, -0.009] 10% * |
| BBBP | B | 0.915 | -0.011 [-0.019, -0.005] 13% * | -0.004 [-0.012, -0.001] 17% * | -0.008 [-0.013, -0.001] 27% * | -0.011 [-0.021, -0.007] 10% * | -0.006 [-0.012, -0.000] 30% * |
| BACE | A | 0.814 | -0.004 [-0.010, +0.000] 33% | -0.001 [-0.008, +0.004] 47% | +0.000 [-0.005, +0.004] 53% | +0.001 [-0.005, +0.006] 53% | +0.005 [-0.003, +0.009] 63% |
| BACE | B | 0.814 | -0.001 [-0.006, +0.008] 47% | +0.004 [-0.006, +0.009] 53% | +0.004 [-0.002, +0.008] 57% | -0.003 [-0.005, +0.009] 47% | +0.005 [+0.000, +0.014] 67% |

Secondary metrics (own Holm families; full table in 1_tuning_vs_default.csv). Count of optimizer x dataset x experiment cells by conclusion. PR-AUC test prevalence (chance level): BBBP 0.740, BACE 0.342.

| metric | better | n.s. | worse |
| --- | --- | --- | --- |
| MAE | 20 | 0 | 10 |
| PR-AUC | 0 | 13 | 7 |
| R2 | 20 | 0 | 10 |

## 2. Optimizers (default excluded; primary metric)

| dataset | exp | Friedman chi2 | p | rank Random | rank TPE | rank GWO | rank WOA | rank PSO | Holm p<.05 pairs | smallest Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESOL | A | 7.63 | 0.106 | 3.50 | 3.13 | 2.70 | 2.50 | 3.17 | 0/10 | Random vs GWO: 0.803 (A12 0.39) |
| ESOL | B | 6.40 | 0.171 | 3.37 | 2.50 | 3.07 | 3.30 | 2.77 | 0/10 | TPE vs WOA: 0.113 (A12 0.64) |
| FreeSolv | A | 13.52 | 0.009 | 3.37 | 3.07 | 2.27 | 3.60 | 2.70 | 2/10 | GWO vs WOA: 0.010 (A12 0.70) |
| FreeSolv | B | 3.39 | 0.495 | 3.07 | 2.87 | 2.77 | 3.43 | 2.87 | 0/10 | GWO vs WOA: 0.767 (A12 0.61) |
| Lipophilicity | A | 5.42 | 0.247 | 3.50 | 3.00 | 2.60 | 2.82 | 3.08 | 0/10 | Random vs TPE: 0.473 (A12 0.35) |
| Lipophilicity | B | 3.28 | 0.512 | 3.28 | 3.20 | 2.63 | 3.02 | 2.87 | 0/10 | Random vs TPE: 1.000 (A12 0.53) |
| BBBP | A | 8.99 | 0.061 | 3.15 | 3.33 | 2.25 | 3.08 | 3.18 | 0/10 | Random vs GWO: 0.066 (A12 0.28) |
| BBBP | B | 11.44 | 0.022 | 3.40 | 2.45 | 2.77 | 3.62 | 2.77 | 0/10 | TPE vs WOA: 0.062 (A12 0.67) |
| BACE | A | 5.68 | 0.225 | 3.48 | 3.12 | 2.80 | 3.02 | 2.58 | 0/10 | Random vs GWO: 0.332 (A12 0.39) |
| BACE | B | 6.26 | 0.181 | 3.23 | 2.93 | 3.17 | 3.27 | 2.40 | 0/10 | Random vs PSO: 0.407 (A12 0.37) |

Secondary metrics: Friedman p < 0.05 in 2/16 cells; Holm-significant pairs 4/160 (2_friedman.csv, 2_pairwise.csv).

## 3. Validation reliability (primary metric)

Cell: median (test - val) / Spearman rho(val, test) over 30 seeds [bootstrap 95% CI].

| dataset | exp | Default gap | Random | TPE | GWO | WOA | PSO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ESOL | A | +0.028 | +0.190 / -0.13 [-0.44, +0.20] | +0.204 / -0.02 [-0.41, +0.37] | +0.207 / -0.11 [-0.51, +0.29] | +0.181 / +0.08 [-0.29, +0.42] | +0.195 / -0.16 [-0.50, +0.22] |
| ESOL | B | +0.028 | +0.182 / -0.17 [-0.54, +0.22] | +0.182 / +0.19 [-0.23, +0.55] | +0.204 / -0.04 [-0.40, +0.36] | +0.188 / +0.05 [-0.35, +0.39] | +0.187 / -0.16 [-0.51, +0.24] |
| FreeSolv | A | -0.462 | -0.091 / +0.10 [-0.34, +0.48] | +0.117 / +0.17 [-0.20, +0.49] | -0.077 / +0.42 [+0.07, +0.72] | +0.061 / +0.23 [-0.20, +0.62] | -0.134 / +0.13 [-0.29, +0.52] |
| FreeSolv | B | -0.462 | -0.003 / +0.05 [-0.31, +0.42] | +0.371 / +0.24 [-0.12, +0.55] | +0.372 / +0.28 [-0.10, +0.58] | +0.144 / -0.32 [-0.66, +0.08] | +0.165 / -0.19 [-0.51, +0.20] |
| Lipophilicity | A | +0.023 | +0.031 / +0.04 [-0.32, +0.37] | +0.032 / -0.09 [-0.44, +0.29] | +0.037 / -0.28 [-0.61, +0.14] | +0.030 / +0.00 [-0.38, +0.38] | +0.035 / +0.26 [-0.15, +0.62] |
| Lipophilicity | B | +0.023 | +0.028 / +0.22 [-0.15, +0.53] | +0.043 / +0.05 [-0.30, +0.38] | +0.038 / +0.39 [+0.04, +0.67] | +0.035 / +0.32 [-0.12, +0.70] | +0.038 / +0.33 [-0.08, +0.65] |
| BBBP | A | -0.009 | -0.040 / -0.42 [-0.67, -0.08] | -0.047 / -0.28 [-0.63, +0.14] | -0.038 / -0.00 [-0.38, +0.38] | -0.046 / -0.20 [-0.57, +0.21] | -0.044 / -0.03 [-0.39, +0.35] |
| BBBP | B | -0.009 | -0.043 / +0.07 [-0.26, +0.39] | -0.042 / +0.17 [-0.25, +0.57] | -0.046 / -0.09 [-0.43, +0.27] | -0.046 / -0.24 [-0.59, +0.18] | -0.040 / +0.05 [-0.29, +0.38] |
| BACE | A | -0.045 | -0.073 / -0.45 [-0.73, -0.10] | -0.074 / +0.09 [-0.29, +0.45] | -0.070 / -0.16 [-0.54, +0.28] | -0.070 / +0.02 [-0.31, +0.37] | -0.068 / -0.14 [-0.51, +0.27] |
| BACE | B | -0.045 | -0.077 / -0.07 [-0.49, +0.38] | -0.083 / +0.01 [-0.38, +0.42] | -0.090 / -0.11 [-0.44, +0.24] | -0.077 / -0.05 [-0.41, +0.32] | -0.075 / +0.12 [-0.27, +0.48] |

## 4. Experiment B feature selection

Cell: median selected fraction [bootstrap 95% CI] / Nogueira stability [95% CI].

| dataset | F | Random | TPE | GWO | WOA | PSO |
| --- | --- | --- | --- | --- | --- | --- |
| ESOL | 188 | 0.52 [0.49, 0.53] / 0.016 [0.005, 0.027] | 0.52 [0.50, 0.52] / 0.015 [0.006, 0.024] | 0.45 [0.43, 0.47] / 0.017 [0.008, 0.026] | 0.53 [0.48, 0.65] / -0.001 [-0.010, 0.009] | 0.51 [0.49, 0.53] / 0.013 [0.004, 0.023] |
| FreeSolv | 171 | 0.51 [0.49, 0.54] / 0.024 [0.012, 0.035] | 0.49 [0.47, 0.50] / 0.026 [0.017, 0.036] | 0.43 [0.42, 0.44] / 0.026 [0.014, 0.037] | 0.58 [0.41, 0.68] / -0.001 [-0.012, 0.011] | 0.49 [0.47, 0.51] / 0.017 [0.008, 0.027] |
| Lipophilicity | 207 | 0.52 [0.51, 0.53] / 0.010 [0.002, 0.017] | 0.51 [0.50, 0.53] / 0.018 [0.009, 0.027] | 0.47 [0.46, 0.48] / 0.014 [0.008, 0.020] | 0.59 [0.55, 0.67] / -0.001 [-0.011, 0.009] | 0.52 [0.50, 0.54] / 0.017 [0.008, 0.026] |
| BBBP | 210 | 0.51 [0.49, 0.52] / 0.007 [-0.002, 0.016] | 0.49 [0.48, 0.52] / 0.011 [0.001, 0.021] | 0.44 [0.42, 0.45] / 0.019 [0.008, 0.029] | 0.50 [0.45, 0.56] / 0.003 [-0.007, 0.012] | 0.50 [0.49, 0.50] / 0.011 [0.001, 0.020] |
| BACE | 193 | 0.49 [0.48, 0.51] / 0.012 [0.002, 0.022] | 0.50 [0.48, 0.53] / 0.010 [-0.003, 0.023] | 0.44 [0.43, 0.46] / 0.033 [0.019, 0.047] | 0.43 [0.41, 0.48] / 0.005 [-0.004, 0.014] | 0.50 [0.48, 0.51] / 0.021 [0.011, 0.032] |

## 5. Mirror check (included as-is from MIRROR_EVAL.md)

```text
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
```

## 6. Sensitivity: a0 schedule vs reference schedule (primary metric, paired by seed)

median paired diff > 0 = reference variant better; Holm over the 10 dataset x experiment cells, per method.

| method | dataset | exp | median ref | median a0 | paired diff | p Holm | unique ref / a0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GWO | ESOL | A | 0.9163 | 0.9134 | -0.0023 | 1.000 | 100 / 91 |
| GWO | ESOL | B | 0.9188 | 0.9110 | -0.0081 | 0.417 | 100 / 91 |
| GWO | FreeSolv | A | 2.4558 | 2.5143 | +0.0534 | 1.000 | 100 / 91 |
| GWO | FreeSolv | B | 2.4255 | 2.3973 | -0.0317 | 1.000 | 100 / 91 |
| GWO | Lipophilicity | A | 0.7262 | 0.7263 | +0.0002 | 1.000 | 100 / 91 |
| GWO | Lipophilicity | B | 0.7340 | 0.7377 | +0.0059 | 1.000 | 100 / 91 |
| GWO | BBBP | A | 0.9095 | 0.9075 | +0.0022 | 1.000 | 100 / 91 |
| GWO | BBBP | B | 0.9070 | 0.9070 | +0.0036 | 1.000 | 100 / 91 |
| GWO | BACE | A | 0.8146 | 0.8152 | -0.0020 | 1.000 | 100 / 91 |
| GWO | BACE | B | 0.8186 | 0.8143 | +0.0029 | 1.000 | 100 / 91 |
| WOA | ESOL | A | 0.9110 | 0.9047 | -0.0032 | 1.000 | 98 / 92 |
| WOA | ESOL | B | 0.9179 | 0.9109 | -0.0034 | 1.000 | 98 / 92 |
| WOA | FreeSolv | A | 2.6027 | 2.5907 | -0.0060 | 1.000 | 96 / 92 |
| WOA | FreeSolv | B | 2.5442 | 2.4826 | -0.0380 | 1.000 | 97 / 92 |
| WOA | Lipophilicity | A | 0.7273 | 0.7272 | -0.0007 | 1.000 | 97 / 93 |
| WOA | Lipophilicity | B | 0.7392 | 0.7455 | +0.0060 | 1.000 | 98 / 92 |
| WOA | BBBP | A | 0.9035 | 0.9021 | -0.0001 | 1.000 | 96 / 92 |
| WOA | BBBP | B | 0.9037 | 0.9026 | -0.0009 | 1.000 | 97 / 92 |
| WOA | BACE | A | 0.8158 | 0.8152 | -0.0052 | 1.000 | 96 / 92 |
| WOA | BACE | B | 0.8115 | 0.8130 | -0.0004 | 1.000 | 98 / 93 |

RQ1 (tuning vs default) conclusions with a0 replacing ref: 0 of 50 optimizer x cell conclusions change.

## 7. Supplement: CD diagrams across datasets (optimizers only; LOW-POWERED, N = 5)

| exp | Random | TPE | GWO | WOA | PSO | CD | Friedman p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 4.60 | 3.40 | 2.00 | 2.40 | 2.60 | 2.73 | 0.075 |
| B | 3.80 | 2.60 | 2.40 | 4.60 | 1.60 | 2.73 | 0.023 |

Figure: figS_cd_optimizers.pdf, figS_cd_optimizers.png, figS_cd_optimizers.tiff

## 8. EXPLORATORY (not pre-specified): validation headroom vs test gain

Headroom = median best validation score of the 5 optimizers relative to the default's validation score, oriented so > 0 = optimizers better on validation (for RMSE: default - optimizers; for ROC-AUC: optimizers - default). Test gain = median test improvement over default (same orientation). % = relative to the default's value.

| dataset | exp | val headroom | val headroom % | test gain | test gain % |
| --- | --- | --- | --- | --- | --- |
| ESOL | A | +0.1385 | +16.1 | -0.0276 | -3.1 |
| ESOL | B | +0.1357 | +15.8 | -0.0206 | -2.3 |
| FreeSolv | A | +0.7031 | +21.5 | +0.2930 | +10.4 |
| FreeSolv | B | +1.0026 | +30.6 | +0.3555 | +12.6 |
| Lipophilicity | A | +0.0426 | +5.8 | +0.0330 | +4.3 |
| Lipophilicity | B | +0.0348 | +4.7 | +0.0230 | +3.0 |
| BBBP | A | +0.0217 | +2.3 | -0.0117 | -1.3 |
| BBBP | B | +0.0261 | +2.8 | -0.0096 | -1.0 |
| BACE | A | +0.0270 | +3.1 | +0.0001 | +0.0 |
| BACE | B | +0.0397 | +4.6 | +0.0029 | +0.4 |

Spearman over the 10 cells: relative (%) rho = +0.41 (p = 0.244); raw units rho = +0.39 (p = 0.260). EXPLORATORY; 10 cells; no correction.

## 9. Compute

Calls with wall time > 600 s (machine sleep) dropped: 10 calls in 10 runs. Total run time, all primary runs: 200.2 worker-hours after dropping (228.6 h raw). Time per call measured under 10 concurrent workers.

| dataset | Random s/call | TPE s/call | GWO s/call | WOA s/call | PSO s/call | Random unique | TPE unique | GWO unique | WOA unique | PSO unique | worker-h (all methods) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ESOL | 2.01 | 3.57 | 4.02 | 2.89 | 3.27 | 100 | 100 | 100 | 98 | 100 | 39.6 |
| FreeSolv | 1.23 | 0.76 | 0.70 | 0.89 | 0.90 | 100 | 100 | 100 | 97 | 100 | 11.4 |
| Lipophilicity | 8.68 | 11.01 | 10.17 | 10.21 | 11.11 | 100 | 100 | 100 | 97 | 100 | 119.8 |
| BBBP | 0.92 | 1.47 | 1.98 | 1.36 | 1.43 | 100 | 100 | 100 | 97 | 100 | 14.4 |
| BACE | 1.38 | 1.08 | 1.28 | 1.16 | 1.35 | 100 | 100 | 100 | 97 | 100 | 15.0 |
