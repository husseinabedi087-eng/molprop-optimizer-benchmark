# xgb_mh TODO

## GWO / WOA variants (decided 2026-09-27)

- `gwo_ref`, `woa_ref` (optimizers/gwo_ref.py, woa_ref.py): reference-faithful schedule
  a = 2 - t*(2/T) (WOA also a2 = -1 + t*(-1/T)), evaluated updates t = 0..8 of T = 10, so a never reaches 0.
  These are the PRIMARY GWO/WOA results, whatever their performance. Run folder: results/xgb_mh/full_ref.
- `gwo`, `woa` (optimizers/gwo.py, woa.py), reported as `gwo_a0`, `woa_a0`: schedule reaching a = 0 on the last
  update. Finished in results/xgb_mh/full. Sensitivity analysis only. At a = 0, gwo_a0 collapses the pack onto
  one point (calls 92-100 are cache hits, effective budget 91); woa_a0 sends p < 0.5 whales exactly onto X*.
  Code of that run is frozen in results/xgb_mh/code_snapshots/full_4669997af4a6/ (hash verified).

## Status (2026-09-28)

- full (gwo_a0/woa_a0 + random/tpe/pso) complete; code frozen in code_snapshots/full_4669997af4a6/.
- full_ref (gwo_ref/woa_ref) complete, audit passed; code frozen in code_snapshots/full_ref_f056a88816b3/.
- Mirror check implemented (`full_run.py --mirror`, `space.mirror`), prediction pre-registered in
  results/xgb_mh/mirror/prediction.md (sha256 in prediction.sha256). Launch: run_xgb_mh_mirror.ps1.
- Wall-clock caveat for the timing table: 13 runs in full and 12 in full_ref include machine suspend time
  (e.g. one FreeSolv woa_ref call of 10,226 s). Report medians, or drop calls with wall_time > ~10 min.

## Later robustness check: mirrored encoding (GWO bias toward x = 0) - uses gwo_ref

Sphere sanity check (results/xgb_mh/sanity/sphere_corner_bias.csv): GWO does worse at d = 200 when the optimum
is at 0.8 than at 0.2 (gwo_a0 1.46x, gwo_ref 1.60x; Random, TPE, PSO ~1.0). In Experiment B, x -> 0 means
deselected features and low hyperparameter values, so this may interact with the feature-count penalty
(GWO selects the fewest features on 4 of 5 datasets in the main run).

Plan (not part of the main runs): rerun Experiment B with a mirrored encoding, decode(1 - x), on 1-2 datasets
(e.g. ESOL and BBBP), for ALL optimizers, same seeds/budget, with **gwo_ref** (and woa_ref) as the GWO/WOA
variants - not gwo_a0/woa_a0. Compare n_selected and test metrics with the unmirrored runs; if GWO's ranking or
feature count shifts materially, report it.
