# Pre-registered prediction: mirrored-encoding robustness check

Recorded 2026-09-28, before the mirror run was implemented or run (no mirrored result exists yet).

## Design

- Mirrored encoding: every point x proposed by an optimizer is mapped to 1 - x (all coordinates) before
  decode(); the common 10-point initial design is mirrored the same way. Feature j is therefore selected
  when x_j < 0.5 instead of x_j > 0.5, and every hyperparameter coordinate is reversed.
- Scope: Experiment B only; BBBP and FreeSolv; optimizers random, tpe, gwo_ref, woa_ref, pso; tuning
  seeds 0-29; budget 100; output results/xgb_mh/mirror/ (300 runs).
- Normal-encoding comparison runs (paired by dataset, optimizer and seed):
  random, tpe, pso from results/xgb_mh/full; gwo_ref, woa_ref from results/xgb_mh/full_ref.

## Prediction

"If GWO's bias toward x = 0 drives its low feature counts, gwo_ref under the mirrored encoding will select
MORE features than under the normal encoding (paired by seed), and its selected fraction will move above
0.5. Random's feature-count distribution should not change."

## Pre-specified test

Per dataset and optimizer: paired Wilcoxon signed-rank test, normal vs mirrored encoding, pairs = the 30
tuning seeds, on
1. the selected-feature fraction (n_selected / F), and
2. the test primary metric (RMSE for FreeSolv, ROC-AUC for BBBP).

Holm correction over the 5 optimizers, applied separately within each dataset x outcome
(4 families of 5 tests).

Reading of the result for the prediction:
- gwo_ref supports it if the mirrored fraction is higher (Holm-adjusted p < 0.05) on each dataset AND its
  median mirrored fraction is > 0.5.
- random supports it if its fraction shows no Holm-significant difference on either dataset.
The test-metric comparison is reported alongside; no directional prediction is made for it.
