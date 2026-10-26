"""J2 (validation-reliability control, R2.2) and J6 (independent scaffold partitions, R2.1), as pre-registered."""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
OUT = ROOT / "analysis" / "revision" / "out"; RV = ROOT / "results" / "revision"
B = 10000; rng = np.random.default_rng(20261025)

# ======================= J2
P = pd.DataFrame([json.loads(f.read_text()) for f in (RV / "J2").glob("*.json")])
res, sel = [], []
for ds in fs.DATASETS:
    cls = fs.TASK[ds] == "classification"
    d = P[(P.dataset == ds) & (P.k >= 0)].sort_values("k"); dflt = P[(P.dataset == ds) & (P.k < 0)].iloc[0]
    sv, cv, tp, ts, tc = (d[c].values for c in ("single_val", "cv_mean", "test_pre", "test_refit_single", "test_refit_cv"))  # all lower = better
    n = len(d); top = np.argsort(sv)[:20]; topc = np.argsort(cv)[:20]
    rho = lambda a, b: stats.spearmanr(a, b)[0]
    idx = rng.integers(0, n, (2000, n))
    def boot(f):
        v = np.array([f(i) for i in idx]); return np.nanpercentile(v, [2.5, 97.5])
    def topk(crit, test, i):
        i = np.asarray(i); o = i[np.argsort(crit[i])[:20]]; return rho(crit[o], test[o])
    r = {"dataset": ds, "rho_single_pre_all": rho(sv, tp), "rho_single_pre_top20": rho(sv[top], tp[top]),
         "rho_single_refit_all": rho(sv, ts), "rho_cv_refit_all": rho(cv, tc), "rho_cv_refit_top20": rho(cv[topc], tc[topc]),
         "rho_single_refit_top20": rho(sv[top], ts[top])}
    lo, hi = boot(lambda i: rho(sv[i], tp[i]) - topk(sv, tp, i)); r.update({"P1_diff_all_minus_top20": r["rho_single_pre_all"] - r["rho_single_pre_top20"], "P1_lo": lo, "P1_hi": hi})
    lo, hi = boot(lambda i: rho(sv[i], tp[i]) - rho(sv[i], ts[i])); r.update({"P2_pre_minus_refit": r["rho_single_pre_all"] - r["rho_single_refit_all"], "P2_lo": lo, "P2_hi": hi})
    lo, hi = boot(lambda i: rho(cv[i], tc[i]) - rho(sv[i], ts[i])); r.update({"P3_cv_minus_single_all": r["rho_cv_refit_all"] - r["rho_single_refit_all"], "P3_lo": lo, "P3_hi": hi})
    lo, hi = boot(lambda i: topk(cv, tc, i) - topk(sv, ts, i)); r.update({"P3_cv_minus_single_top20": r["rho_cv_refit_top20"] - r["rho_single_refit_top20"], "P3t_lo": lo, "P3t_hi": hi})
    # P4 simulated searches
    reg_s, reg_c, beat_s, beat_c, out_s, out_c = [], [], [], [], [], []
    for _ in range(2000):
        i = rng.choice(n, 100, replace=False)
        bs, bc = i[np.argmin(sv[i])], i[np.argmin(cv[i])]
        best_s, best_c = ts[i].min(), tc[i].min()
        reg_s.append(ts[bs] - best_s); reg_c.append(tc[bc] - best_c); out_s.append(ts[bs]); out_c.append(tc[bc])
        beat_s.append(ts[bs] < dflt.test_refit_single); beat_c.append(tc[bc] < dflt.test_refit_single)
    dr = np.array(reg_s) - np.array(reg_c)
    bi = rng.integers(0, 2000, (B, 2000)); bm = dr[bi].mean(1)
    sgn = (lambda v: 1 - v) if cls else (lambda v: v)
    r.update({"P4_regret_single": np.mean(reg_s), "P4_regret_cv": np.mean(reg_c), "P4_diff": dr.mean(), "P4_lo": np.percentile(bm, 2.5), "P4_hi": np.percentile(bm, 97.5),
              "P4_cv_improves": bool(np.percentile(bm, 2.5) > 0), "P4_median_test_single": float(sgn(np.median(out_s))), "P4_median_test_cv": float(sgn(np.median(out_c))),
              "P4_share_beat_default_single": np.mean(beat_s), "P4_share_beat_default_cv": np.mean(beat_c),
              "default_test": float(sgn(dflt.test_refit_single)), "pool_best_test": float(sgn(min(ts.min(), tc.min()))),
              "single_val_range_all": sv.max() - sv.min(), "single_val_range_top20": sv[top].max() - sv[top].min()})
    res.append(r)
J2R = pd.DataFrame(res); J2R.to_csv(OUT / "J2_validation_control.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(J2R.round(3).T.to_string())

# ======================= J6
Q = pd.DataFrame([json.loads(f.read_text()) for f in (RV / "J6").glob("*.json")])
NAME = {"random": "Random", "tpe": "TPE", "gwo_ref": "GWO", "woa_ref": "WOA", "pso": "PSO", "default": "Default"}
Q["method"] = Q.optimizer.map(NAME); Q["cls"] = Q.dataset.isin(["BBBP", "BACE"])
Q["test"] = np.where(Q.cls, Q.test_roc_auc, Q.test_rmse); Q["good"] = np.where(Q.cls, Q.test, -Q.test)
out6, pair6, tvd6 = [], [], []
for ds in ("ESOL", "FreeSolv", "BBBP", "BACE"):
    cls = ds in ("BBBP", "BACE")
    T = Q[(Q.dataset == ds) & (Q.method != "Default")]
    parts = sorted(T.partition.unique()); P_, K, S_ = len(parts), 5, 3
    Y = np.zeros((P_, K, S_))
    for a, p in enumerate(parts):
        for b, m in enumerate(fs.OPTS):
            Y[a, b] = T[(T.partition == p) & (T.method == m)].sort_values("seed").good.values
    gm = Y.mean(); pm = Y.mean((1, 2)); om = Y.mean((0, 2)); cm = Y.mean(2)
    ss_p = K * S_ * ((pm - gm) ** 2).sum(); ss_o = P_ * S_ * ((om - gm) ** 2).sum()
    ss_po = S_ * ((cm - pm[:, None] - om[None, :] + gm) ** 2).sum(); ss_e = ((Y - cm[:, :, None]) ** 2).sum()
    df_p, df_o, df_po, df_e = P_ - 1, K - 1, (P_ - 1) * (K - 1), P_ * K * (S_ - 1)
    ms_p, ms_o, ms_po, ms_e = ss_p / df_p, ss_o / df_o, ss_po / df_po, ss_e / df_e
    F = ms_o / ms_po; pF = stats.f.sf(F, df_o, df_po)
    var_e = ms_e; var_po = max((ms_po - ms_e) / S_, 0); var_p = max((ms_p - ms_po) / (K * S_), 0)
    dflt = Q[(Q.dataset == ds) & (Q.method == "Default")].set_index("partition").loc[parts]
    delta = (0.01 * np.ones(P_)) if cls else 0.02 * dflt.test_rmse.values
    out6.append({"dataset": ds, "F_optimizer": F, "p_optimizer": pF, "var_partition": var_p, "var_opt_x_partition": var_po, "var_tuning_seed": var_e,
                 "partition_gt_seed": var_p > var_e, "friedman_on_partition_means_p": stats.friedmanchisquare(*cm.T).pvalue,
                 "n_train_range": f"{T.n_train.min()}-{T.n_train.max()}", "F_range": f"{T.F.min()}-{T.F.max()}",
                 **{f"mean_{m}": om[i] * (1 if cls else -1) for i, m in enumerate(fs.OPTS)}, "default_mean": dflt.test.mean()})
    for i, j in itertools.combinations(range(K), 2):
        d = cm[:, i] - cm[:, j]; mu = d.mean(); se = d.std(ddof=1) / np.sqrt(P_); dl = delta.mean()
        lo95, hi95 = mu + np.array([-1, 1]) * stats.t.ppf(.975, P_ - 1) * se
        pt = max(stats.t.sf((mu + dl) / se, P_ - 1), stats.t.cdf((mu - dl) / se, P_ - 1))
        pair6.append({"dataset": ds, "pair": f"{fs.OPTS[i]} vs {fs.OPTS[j]}", "mean_diff_goodness": mu, "ci95_low": lo95, "ci95_high": hi95,
                      "p_t": 2 * stats.t.sf(abs(mu / se), P_ - 1), "delta": dl, "tost_p": pt,
                      "min_equiv_margin": abs(mu) + stats.t.ppf(.95, P_ - 1) * se})
    for b, m in enumerate(fs.OPTS):
        imp = cm[:, b] - dflt.good.values if "good" in dflt else cm[:, b] - np.where(cls, dflt.test.values, -dflt.test.values)
        tvd6.append({"dataset": ds, "method": m, **{f"impr_p{p}": v for p, v in zip(parts, imp)}, "n_partitions_better": int((imp > 0).sum()),
                     "mean_impr": imp.mean()})
O6 = pd.DataFrame(out6); O6.to_csv(OUT / "J6_variance_components.csv", index=False); print(O6.round(5).T.to_string())
PW = pd.DataFrame(pair6)
for ds in PW.dataset.unique():
    m = PW.dataset == ds; PW.loc[m, "p_t_holm"] = fs.holm(PW[m].p_t.tolist()); PW.loc[m, "tost_p_holm"] = fs.holm(PW[m].tost_p.tolist())
PW["equivalent"] = PW.tost_p_holm < .05; PW.to_csv(OUT / "J6_pairwise.csv", index=False)
print(PW.groupby("dataset").agg(n_sig=("p_t_holm", lambda v: int((v < .05).sum())), n_equiv=("equivalent", "sum"), max_margin=("min_equiv_margin", "max"), delta=("delta", "first")))
TV = pd.DataFrame(tvd6); TV.to_csv(OUT / "J6_tuning_vs_default.csv", index=False); print(TV.round(4).to_string())
