# molprop-optimizer-benchmark

Metaheuristic hyperparameter optimisation and joint feature selection for XGBoost on MoleculeNet
molecular-property benchmarks. Five optimizers — Random Search, Optuna TPE, Grey Wolf Optimizer (GWO),
Whale Optimization Algorithm (WOA) and Particle Swarm Optimization (PSO) — are compared with the
XGBoost defaults under a matched budget of 100 fitness evaluations, 30 tuning seeds each.

- **Datasets:** ESOL, FreeSolv, Lipophilicity (regression, RMSE); BBBP, BACE (classification, ROC-AUC).
  Bemis–Murcko scaffold split 80/10/10, split seed 42. Features: RDKit 2D descriptors
  (constant / > 20 % missing features dropped using the training split only).
- **Experiment A:** hyperparameter optimisation only (8-dimensional unit hypercube, one shared `decode()`).
- **Experiment B:** joint feature selection + hyperparameter optimisation (8 + F dimensions; feature j selected
  if x_j > 0.5; fitness = validation metric + 0.01 · selected fraction).
- **Protocol:** validation-only search (the test split is never seen during search, verified structurally),
  shared 10-point initial design for every optimizer, refit on train + validation with the early-stopped tree
  count, single test evaluation.

## Main findings (see `analysis/final/RESULTS.md`)

- Tuning beats the defaults on FreeSolv and Lipophilicity (83–100 % of seeds), is neutral on BACE and
  is worse than the defaults on ESOL and BBBP (Holm-corrected one-sample Wilcoxon).
- The five optimizers are largely indistinguishable: 2 of 100 pairwise comparisons survive Holm correction.
- Best validation score does not predict test score across seeds (46 of 50 Spearman-ρ 95 % CIs include 0).
- Selected feature subsets are essentially unstable (Nogueira Φ between −0.001 and 0.033).
- GWO's low feature counts come from its bias toward x = 0: under a mirrored encoding they flip above 0.5
  (pre-registered prediction in `results/xgb_mh/mirror/prediction.md`, supported).

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Pinned versions (as used for every result here): Python 3.12.14, numpy 2.5.3, pandas 3.0.1, scipy 1.18.1,
scikit-learn 1.9.1, xgboost 3.4.1, optuna 5.0.0, rdkit 2026.3.6, torch 2.14.0, torch-geometric 2.8.0.post1,
joblib 1.6.0, matplotlib 3.11.2, pillow 12.3.0, psutil 7.2.2. CPU only.

Run every command below from the repository root. On Linux/macOS use the `python -m ...` equivalents given
for the PowerShell launchers.

## Reproduce, in order

Runtimes measured on an Intel i7-10750H (6 cores / 12 threads, 16 GB RAM) with 10 parallel workers,
`nthread = 1` per XGBoost model.

| # | Step | Command | Runtime |
|---|---|---|---|
| 1 | Download MoleculeNet, build scaffold splits and RDKit descriptor matrices | `python -m xgb_mh.prepare` | ~1 min + download |
| 2 | Unit tests (decoder, optimizers, mirrored encoding) | `python -m xgb_mh.tests.test_space`<br>`python -m xgb_mh.tests.test_optimizers`<br>`python -m xgb_mh.tests.test_mirror` ¹ | ~1 min, ~1 min, ~2 min |
| 3 | Sphere sanity checks (centered / shifted, corner bias) | `python -m xgb_mh.sanity_sphere`<br>`python -m xgb_mh.sanity_corner` | ~2 min each |
| 4 | Smoke test + runtime estimate (ESOL, BBBP, 2 seeds, budget 20) | `python -m xgb_mh.smoke` | ~11 min |
| 5 | Main run: Random, TPE, PSO (+ GWO / WOA with the `a0` schedule, sensitivity only) | `powershell -File run_xgb_mh_full.ps1`<br>(= `python -m xgb_mh.full_run`) | ~20 h |
| 6 | Reference-faithful GWO / WOA (primary GWO / WOA results) | `powershell -File run_xgb_mh_full_ref.ps1`<br>(= `python -m xgb_mh.full_run --out-dir results/xgb_mh/full_ref --optimizers gwo_ref,woa_ref --datasets BBBP,FreeSolv,BACE,ESOL,Lipophilicity`) | ~8 h |
| 7 | Mirrored-encoding check (pre-registered in `results/xgb_mh/mirror/prediction.md`) | `powershell -File run_xgb_mh_mirror.ps1`<br>(= `python -m xgb_mh.full_run --out-dir results/xgb_mh/mirror --mirror --experiments B --datasets BBBP,FreeSolv --optimizers random,tpe,gwo_ref,woa_ref,pso`) | ~1 h |
| 8 | Integrity audits and pre-registered mirror evaluation ¹ | `python analysis_preview/audit.py --datasets BBBP,FreeSolv,BACE,ESOL,Lipophilicity`<br>`python analysis_preview/audit.py --run-dir results/xgb_mh/full_ref --optimizers gwo_ref,woa_ref --datasets BBBP,FreeSolv,BACE,ESOL,Lipophilicity --design-ref results/xgb_mh/full`<br>`python analysis_preview/mirror_eval.py` | ~1 min each |
| 9 | Final statistics → `analysis/final/` ¹ | `python analysis/final/final_stats.py` | ~30 s |
| 10 | Figures → `analysis/figures/out/` ¹ | `python analysis/figures/make_figures.py` | ~30 s |

¹ Needs the per-call logs (`results/xgb_mh/*/calls/*.jsonl`). They are included as a compressed archive,
`data/xgb_mh_call_logs.zip` (33.4 MB; 2,440 logs, 160 MB unpacked; SHA-256 in `data/xgb_mh_call_logs.zip.sha256`).
Unzip it into the repository root to restore `results/xgb_mh/*/calls/` (it also writes `MANIFEST_call_logs.csv`
with the SHA-256 of every log):

```bash
sha256sum -c data/xgb_mh_call_logs.zip.sha256   # run inside data/, or check the hash manually
unzip data/xgb_mh_call_logs.zip
```

Alternatively, regenerate them with steps 5–7.

The runs (steps 5–7) are resumable: stop them at any time and rerun the same command; finished runs are
skipped and partial runs are redone. Each run folder records the SHA-256 of the result-determining code in
`run_metadata.json` and refuses to resume if that code changes. Wall-clock times in `full_run.log` include
some machine-sleep periods; the compute table in `RESULTS.md` (section 9) drops calls longer than 600 s.

Every step writes into this repository's `results/` and `analysis/` folders, overwriting the included outputs.

Notes on specific files:

- `xgb_mh/sanity_corner.py` was added after the experiments to reproduce `results/xgb_mh/sanity/sphere_corner_bias.csv`,
  originally produced by inline code; verified to reproduce it exactly. It is not part of the hashed
  result-determining code, so the recorded code hashes are unaffected.
- `results/xgb_mh/smoke/` was produced before the shared initial design was applied to TPE and is used only for
  runtime estimates (the resumable runner's ETA), not for any reported result.
- Absolute local paths were replaced with relative ones in two published copies:
  `results/xgb_mh/features/summary.json` and `results/xgb_mh/full/full_run.log`.
- `.gitattributes` disables line-ending conversion so every file keeps the exact bytes behind the SHA-256 values
  recorded in `run_metadata.json`, `prediction.sha256` and `final_stats.json`.

## Folder map

```
xgb_mh/                       study code: data + features, shared decode(), budgeted objective,
  optimizers/                 random_search, tpe, gwo (a0), woa (a0), gwo_ref, woa_ref, pso
  tests/                      unit tests
  full_run.py                 resumable parallel runner          smoke.py, sanity_*.py  checks
run_xgb_mh_*.ps1              launchers for steps 5-7
data/xgb_mh_call_logs.zip     per-call logs of every run (+ .sha256); unzip into the repo root
results/xgb_mh/
  full/  full_ref/  mirror/   test_summary.csv (one row per run), run_metadata.json, full_run.log,
                              runs/*.json (per-run summaries), baseline/*.json
  mirror/prediction.md(.sha256)   pre-registered prediction, hashed before the run
  features/                   descriptor logs + summary (feature matrices are rebuilt by step 1)
  splits/                     scaffold-split indices        sanity/  checks/  smoke/  small check outputs
  code_snapshots/             exact code of the full and full_ref runs (hashes verified)
analysis_preview/             audit and preview scripts; output/mirror_20260928_111045/ is the
                              pre-registered mirror evaluation read by final_stats.py
analysis/final/               final_stats.py, RESULTS.md, final_stats.json, one CSV per analysis,
  code_snapshot/              exact analysis code behind final_stats.json (hashes verified)
analysis/figures/             figure pipeline (style, data, figures, checks); out/ = PDF + PNG + contact sheet
```

Naming: `gwo` / `woa` in `results/xgb_mh/full` are the **a0** variants (coefficient `a` reaches 0 on the last
update; sensitivity analysis only). `gwo_ref` / `woa_ref` follow the reference implementation and are the
primary GWO / WOA results. `xgb_mh/__init__.py` also references split files of an earlier, unrelated study
for an optional cross-check that is skipped when those files are absent.

## Data

MoleculeNet datasets (ESOL, FreeSolv, Lipophilicity, BBBP, BACE) are downloaded by PyTorch Geometric at
step 1 and are not redistributed here. Please cite MoleculeNet (Wu et al., 2018) and the original dataset
sources when using them.

## Citation

```
@misc{molprop_optimizer_benchmark,
  title  = {<TITLE>},
  author = {<AUTHORS>},
  year   = {2026},
  note   = {Code, results and call logs: https://github.com/<USER>/molprop-optimizer-benchmark}
}
```

A Zenodo DOI for the whole repository will be added at the first release.

## License

MIT (see `LICENSE`).
