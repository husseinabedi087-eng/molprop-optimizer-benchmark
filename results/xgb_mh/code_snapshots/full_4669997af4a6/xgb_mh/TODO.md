# xgb_mh TODO

## Later robustness check: mirrored encoding (GWO bias toward x = 0)

Sphere sanity check (results/xgb_mh/sanity/sphere_corner_bias.csv) shows GWO does ~1.46x worse at
d = 200 when the optimum is at 0.8 than at 0.2 (Random, TPE, PSO ~1.0). In Experiment B, x -> 0
means deselected features and low hyperparameter values, so this may interact with the feature-count
penalty.

Plan (not part of the main run): rerun Experiment B with a mirrored encoding, decode(1 - x), on
1-2 datasets (e.g. ESOL and BBBP), for ALL optimizers, same seeds/budget. Compare n_selected and
test metrics with the main run; if GWO's ranking or feature count shifts materially, report it.
