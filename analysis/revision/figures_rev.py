"""Revised manuscript figures: printed at their placed size, all text >= 10 pt, no panel letters, no outer borders."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "final")); sys.path.insert(0, str(ROOT / "analysis" / "figures"))
import final_stats as fs
from style import COLORS, MARKERS
OUT = ROOT / "analysis" / "revision" / "fig"; OUT.mkdir(parents=True, exist_ok=True)
A = ROOT / "analysis" / "revision" / "out"
W = 6.5; FS = 10
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": FS,
                     "axes.labelsize": FS, "axes.titlesize": FS, "legend.fontsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "lines.linewidth": 1.0,
                     "legend.frameon": False, "savefig.dpi": 600, "figure.facecolor": "white", "pdf.fonttype": 42})
OPTS = fs.OPTS; DS = fs.DATASETS
ex = pd.read_csv(A / "exact_runs.csv")
AUDIT = {}

def save(fig, name):
    from matplotlib.text import Text
    fig.canvas.draw()
    AUDIT[name] = {"size_in": [round(float(v), 2) for v in fig.get_size_inches()],
                   "font_sizes": sorted({round(t.get_fontsize(), 1) for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()})}
    fig.savefig(OUT / f"{name}.png", dpi=600); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)

def ylab(ds): return "Test ROC-AUC" if fs.TASK[ds] == "classification" else "Test RMSE"

tvd = fs.tuning_vs_default(ex); tvd = tvd[tvd.role == "primary"]

# ---------------- Fig 2: test distributions
fig, axes = plt.subplots(5, 2, figsize=(W, 4.8), sharey="row")
rng = np.random.default_rng(0)
for i, ds in enumerate(DS):
    for j, e in enumerate(fs.EXPERIMENTS):
        ax = axes[i, j]; d0 = fs.cell(ex, ds, e, "Default", "test_primary")[0]
        vals = [fs.cell(ex, ds, e, m, "test_primary") for m in OPTS]
        bp = ax.boxplot(vals, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
        for k, (b, m) in enumerate(zip(bp["boxes"], OPTS)):
            b.set_facecolor("white"); b.set_edgecolor(COLORS[m])
            ax.scatter(k + 1 + rng.uniform(-.18, .18, len(vals[k])), vals[k], s=4, color=COLORS[m], marker=MARKERS[m], alpha=.6, lw=0)
        ax.axhline(d0, color="black", ls=(0, (4, 2)), lw=0.9)
        top = max(v.max() for v in vals + [np.array([d0])]); bot = min(v.min() for v in vals + [np.array([d0])])
        for k, m in enumerate(OPTS):
            r = tvd[(tvd.dataset == ds) & (tvd.experiment == e) & (tvd.method == m)].iloc[0]
            if r.p_holm < .05: ax.text(k + 1, top + 0.03 * (top - bot), "*", ha="center", va="bottom", fontsize=FS)
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.16 * (top - bot))
        ax.set_xticks(range(1, 6)); ax.set_xticklabels(OPTS if i == 4 else [""] * 5, rotation=0)
        ax.set_title(f"{ds}, Exp. {e}", loc="left")
        if j == 0: ax.set_ylabel(ylab(ds))
fig.tight_layout(h_pad=0.6, w_pad=1.0); save(fig, "fig2_test_distributions")

# ---------------- Fig 3: improvement over default
fig, axes = plt.subplots(5, 2, figsize=(W, 4.5), sharex="row", sharey=True)
for i, ds in enumerate(DS):
    for j, e in enumerate(fs.EXPERIMENTS):
        ax = axes[i, j]
        for k, m in enumerate(OPTS):
            r = tvd[(tvd.dataset == ds) & (tvd.experiment == e) & (tvd.method == m)].iloc[0]
            ax.errorbar(r.median_improvement, k, xerr=[[r.median_improvement - r.impr_ci_low], [r.impr_ci_high - r.median_improvement]],
                        fmt=MARKERS[m], color=COLORS[m], ms=5, capsize=2)
            if r.p_holm < .05: ax.text(r.impr_ci_high, k, " *", va="center", fontsize=FS)
        ax.axvline(0, color="black", lw=0.8); ax.set_yticks([]); ax.set_ylim(4.6, -0.6); ax.locator_params(axis="x", nbins=4)
        ax.set_title(f"{ds}, Exp. {e}", loc="left")
        if i == 4 or True: ax.set_xlabel("Improvement over default" if i == 4 else "")
fig.legend(handles=[Line2D([], [], marker=MARKERS[m], color=COLORS[m], ls="", label=m) for m in OPTS], ncol=5, loc="upper center")
fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.5, w_pad=1.0); save(fig, "fig3_improvement_vs_default")

# ---------------- Fig 4: validation vs test (one panel per dataset)
fig, axes = plt.subplots(2, 3, figsize=(W, 3.6))
for i, ds in enumerate(DS):
    ax = axes.flat[i]
    for m in OPTS:
        for e, fill in (("A", True), ("B", False)):
            v, t = fs.cell(ex, ds, e, m, "val_primary"), fs.cell(ex, ds, e, m, "test_primary")
            ax.scatter(v, t, s=12, marker=MARKERS[m], facecolors=COLORS[m] if fill else "none", edgecolors=COLORS[m], lw=0.7)
    dv, dt = fs.cell(ex, ds, "A", "Default", "val_primary")[0], fs.cell(ex, ds, "A", "Default", "test_primary")[0]
    ax.scatter([dv], [dt], marker="*", s=110, color="black", zorder=5)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.8); ax.set_title(ds, loc="left")
    ax.set_xlabel("Best validation " + ("ROC-AUC" if fs.TASK[ds] == "classification" else "RMSE")); ax.set_ylabel(ylab(ds))
    ax.locator_params(nbins=4)
ax = axes.flat[5]; ax.axis("off")
h = [Line2D([], [], marker=MARKERS[m], color=COLORS[m], ls="", ms=6, label=m) for m in OPTS]
h += [Line2D([], [], marker="o", color="grey", ls="", ms=6, label="filled: Exp. A"), Line2D([], [], marker="o", mfc="none", color="grey", ls="", ms=6, label="open: Exp. B"),
      Line2D([], [], marker="*", color="black", ls="", ms=10, label="Default"), Line2D([], [], color="#999999", label="y = x")]
ax.legend(handles=h, loc="center", ncol=1)
fig.tight_layout(h_pad=1.0, w_pad=0.8); save(fig, "fig4_validation_vs_test")

# ---------------- Fig 5: Spearman rho, refitted (primary) and pre-refit diagnostic
vr = fs.validation_reliability(ex); vr = vr[vr.method != "Default"]
pre = pd.read_csv(A / "J1_prerefit_diagnostic.csv")
fig, axes = plt.subplots(1, 2, figsize=(W, 8.2), sharey=True)
labels = [f"{ds} {m}" for ds in DS for m in OPTS]
for j, e in enumerate(fs.EXPERIMENTS):
    ax = axes[j]; y = 0
    for ds in DS:
        for m in OPTS:
            r = vr[(vr.dataset == ds) & (vr.experiment == e) & (vr.method == m)].iloc[0]
            p = pre[(pre.dataset == ds) & (pre.experiment == e) & (pre.method == m)].iloc[0]
            ax.errorbar(r.rho, y - 0.17, xerr=[[r.rho - r.rho_ci_low], [r.rho_ci_high - r.rho]], fmt=MARKERS[m], color=COLORS[m], ms=4.5, capsize=0, lw=0.9)
            ax.errorbar(p.rho_pre, y + 0.17, xerr=[[p.rho_pre - p.rho_pre_lo], [p.rho_pre_hi - p.rho_pre]], fmt=MARKERS[m], mfc="white", color=COLORS[m], ms=4.5, capsize=0, lw=0.9)
            y += 1
        ax.axhline(y - 0.5, color="#DDDDDD", lw=0.6)
    ax.axvline(0, color="black", lw=0.8); ax.set_xlim(-1, 1); ax.set_xlabel("Spearman ρ (validation, test)"); ax.set_title(f"Exp. {e}", loc="left")
axes[0].set_yticks(range(25)); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], marker="o", color="grey", ls="", label="refitted model (primary)"),
                    Line2D([], [], marker="o", mfc="white", color="grey", ls="", label="pre-refit model (diagnostic)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=1.0); save(fig, "fig5_spearman_rho")

# ---------------- Fig 6: selected fraction, Exp B
fig, ax = plt.subplots(figsize=(W, 3.3))
for i, ds in enumerate(DS):
    for k, m in enumerate(OPTS):
        s = ex[(ex.dataset == ds) & (ex.experiment == "B") & (ex.method == m)]
        fr = (s.n_selected / s.n_features_total).values; x = i * 6 + k
        ax.boxplot([fr], positions=[x], widths=0.7, showfliers=False, medianprops=dict(color=COLORS[m], lw=1.5),
                   boxprops=dict(color=COLORS[m]), whiskerprops=dict(color=COLORS[m]), capprops=dict(color=COLORS[m]))
ax.axhline(0.5, color="#999999", lw=0.8)
ax.set_xticks([i * 6 + 2 for i in range(5)]); ax.set_xticklabels(DS); ax.set_ylabel("Selected fraction")
ax.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label=m) for m in OPTS], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); save(fig, "fig6_selected_fraction")

# ---------------- Fig 7: Nogueira stability
fsel = fs.feature_selection(ex)
fig, ax = plt.subplots(figsize=(W, 3.0))
for i, ds in enumerate(DS):
    for k, m in enumerate(OPTS):
        r = fsel[(fsel.dataset == ds) & (fsel.method == m)].iloc[0]; x = i * 6 + k
        ax.errorbar(x, r.nogueira, yerr=[[r.nogueira - r.nogueira_ci_low], [r.nogueira_ci_high - r.nogueira]], fmt=MARKERS[m], color=COLORS[m], ms=5, capsize=2)
ax.axhline(0, color="black", lw=0.8); ax.set_xticks([i * 6 + 2 for i in range(5)]); ax.set_xticklabels(DS); ax.set_ylabel("Nogueira stability")
ax.legend(handles=[Line2D([], [], marker=MARKERS[m], color=COLORS[m], ls="", label=m) for m in OPTS], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
fig.tight_layout(); save(fig, "fig7_stability")

# ---------------- Fig 8: mirror check, standard / full mirror / descriptor-only mirror
j3 = pd.read_csv(A / "J3_descriptor_mirror.csv")
src = {"none": None, "all": "mirror"}
NAME = {"Random": "random", "TPE": "tpe", "GWO": "gwo_ref", "WOA": "woa_ref", "PSO": "pso"}
FOLD = {"random": "full", "tpe": "full", "pso": "full", "gwo_ref": "full_ref", "woa_ref": "full_ref"}
RES = ROOT / "results" / "xgb_mh"; SR = ROOT / "results" / "revision" / "searches"
fig, axes = plt.subplots(1, 2, figsize=(W, 3.4), sharey=True)
for a, ds in enumerate(("BBBP", "FreeSolv")):
    ax = axes[a]
    for k, m in enumerate(OPTS):
        o = NAME[m]; fr = {}
        for cond, folder in (("S", FOLD[o]), ("M", "mirror")):
            fr[cond] = [json.loads((RES / folder / "runs" / f"{ds}_B_{o}_s{s}.json").read_text()) for s in range(30)]
            fr[cond] = np.array([d["n_selected"] / d["n_features_total"] for d in fr[cond]])
        dd = [json.loads((SR / f"{ds}_B_{o}_s{s}_m-desc_b100_lam0.01abs.json").read_text()) for s in range(30)]
        fr["D"] = np.array([d["n_selected"] / d["n_features_total"] for d in dd])
        for c, (cond, mk) in enumerate((("S", "o"), ("M", "s"), ("D", "D"))):
            v = fr[cond]; x = k + (c - 1) * 0.25; q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color=COLORS[m], lw=1.2); ax.plot(x, q2, marker=mk, color=COLORS[m], mfc="white" if cond != "S" else COLORS[m], ms=5)
    ax.axhline(0.5, color="#999999", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(OPTS); ax.set_title(f"{ds}, Exp. B", loc="left")
axes[0].set_ylabel("Selected fraction")
fig.legend(handles=[Line2D([], [], marker="o", color="grey", ls="", label="standard"), Line2D([], [], marker="s", mfc="white", color="grey", ls="", label="all coordinates mirrored"),
                    Line2D([], [], marker="D", mfc="white", color="grey", ls="", label="descriptor genes mirrored")], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0); save(fig, "fig8_mirror")

# ---------------- Fig 9: validation control (J2)
j2 = pd.read_csv(A / "J2_validation_control.csv").set_index("dataset").loc[list(DS)]
fig, axes = plt.subplots(1, 2, figsize=(W, 3.5))
x = np.arange(5)
ax = axes[0]
ax.bar(x - 0.18, j2.rho_single_refit_all, 0.36, color="#BBBBBB", label="single split")
ax.bar(x + 0.18, j2.rho_cv_refit_all, 0.36, color="#0072B2", label="inner 5-fold CV")
ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["ESOL", "FreeSolv", "Lipo", "BBBP", "BACE"], rotation=35, ha="right")
ax.set_ylabel("Spearman ρ over 200 configs"); ax.set_title("Criterion vs test score", loc="left")
ax = axes[1]
ax.bar(x - 0.18, 100 * j2.P4_share_beat_default_single, 0.36, color="#BBBBBB", label="single split")
ax.bar(x + 0.18, 100 * j2.P4_share_beat_default_cv, 0.36, color="#0072B2", label="inner 5-fold CV")
ax.set_xticks(x); ax.set_xticklabels(["ESOL", "FreeSolv", "Lipo", "BBBP", "BACE"], rotation=35, ha="right"); ax.set_ylim(0, 100)
ax.set_ylabel("Searches beating default (%)"); ax.set_title("Selection by each criterion", loc="left")
fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=["single validation split", "inner 5-fold scaffold CV"], ncol=2, loc="lower center")
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.5); save(fig, "fig9_validation_control")

# ---------------- Fig 1: workflow
from matplotlib.patches import FancyBboxPatch
fig = plt.figure(figsize=(W, 3.05)); ax = fig.add_axes([0.005, 0.005, 0.99, 0.99]); ax.set_xlim(0, 100); ax.set_ylim(0, 51); ax.axis("off")
def box(x, y, w, h, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#333333", lw=0.9))
    ax.text(x + 1.2, y + h - 1.6, title, fontsize=FS, fontweight="bold", va="top")
    ax.text(x + 1.2, y + h - 6.6, "\n".join(lines), fontsize=FS, va="top", linespacing=1.3)
def arrow(x0, y0, x1, y1, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.0, ls=ls, shrinkA=0, shrinkB=0))
Y, H = 16, 28.5
box(0.5, Y, 16.5, H, "1 Data", ["ESOL", "FreeSolv", "Lipophilicity", "BBBP, BACE"])
box(19.5, Y, 19, H, "2 Split, features", ["Scaffold split", "80/10/10, fixed;", "171-210 RDKit", "2D descriptors"])
box(41, Y, 35, H, "3 Matched-budget search", ["Random, TPE, GWO, WOA, PSO", r"Exp. A: $d$ = 8; Exp. B: $d$ = 179-218", "100 calls, 10 shared starts,", "30 seeds, validation fitness"])
box(78.5, Y, 21, H, "4 Refit, test", ["Refit on train +", "validation; one", "test evaluation"])
arrow(17, Y + H / 2, 19.5, Y + H / 2); arrow(38.5, Y + H / 2, 41, Y + H / 2); arrow(76, Y + H / 2, 78.5, Y + H / 2)
ax.plot([29, 29, 89], [Y + H, 48, 48], color="#333333", lw=1.0, ls=(0, (4, 2))); arrow(89, 48, 89, Y + H, ls=(0, (4, 2)))
ax.text(59, 48.6, "Default configuration: all descriptors, no search", fontsize=FS, ha="center", va="bottom")
box(0.5, 0.5, 99, 12, "5 Analyses per dataset and experiment (seed = unit, n = 30)",
    ["Tuning vs default; optimizer differences; validation-test agreement; subsets"])
arrow(89, Y, 89, 12.5)
save(fig, "fig1_workflow")


# ---------------- Fig 5 (combined, no panel letters): selected fraction, stability, mirror
fig = plt.figure(figsize=(W, 4.85))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.8, 1.0], hspace=0.75, wspace=0.12)
ax = fig.add_subplot(gs[0, :])
for i, ds in enumerate(DS):
    for k, m in enumerate(OPTS):
        s_ = ex[(ex.dataset == ds) & (ex.experiment == "B") & (ex.method == m)]
        fr = (s_.n_selected / s_.n_features_total).values; x = i * 6 + k
        ax.boxplot([fr], positions=[x], widths=0.7, showfliers=False, medianprops=dict(color=COLORS[m], lw=1.5),
                   boxprops=dict(color=COLORS[m]), whiskerprops=dict(color=COLORS[m]), capprops=dict(color=COLORS[m]))
ax.axhline(0.5, color="#999999", lw=0.8); ax.set_xticks([i * 6 + 2 for i in range(5)]); ax.set_xticklabels(DS)
ax.set_ylabel("Fraction"); ax.set_title("Selected fraction per run (Exp. B)", loc="left")
ax.legend(handles=[Line2D([], [], color=COLORS[m], lw=2, label=m) for m in OPTS], ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.42))
ax = fig.add_subplot(gs[1, :])
for i, ds in enumerate(DS):
    for k, m in enumerate(OPTS):
        r = fsel[(fsel.dataset == ds) & (fsel.method == m)].iloc[0]; x = i * 6 + k
        ax.errorbar(x, r.nogueira, yerr=[[r.nogueira - r.nogueira_ci_low], [r.nogueira_ci_high - r.nogueira]], fmt=MARKERS[m], color=COLORS[m], ms=4, capsize=1.5)
ax.axhline(0, color="black", lw=0.8); ax.set_xticks([i * 6 + 2 for i in range(5)]); ax.set_xticklabels(DS); ax.set_ylabel("Stability")
ax.set_title("Nogueira stability across 30 seeds", loc="left")
for a_, ds in enumerate(("BBBP", "FreeSolv")):
    ax = fig.add_subplot(gs[2, a_])
    for k, m in enumerate(OPTS):
        o = NAME[m]; fr = {}
        for cond, folder in (("S", FOLD[o]), ("M", "mirror")):
            fr[cond] = np.array([(lambda d: d["n_selected"] / d["n_features_total"])(json.loads((RES / folder / "runs" / f"{ds}_B_{o}_s{s}.json").read_text())) for s in range(30)])
        fr["D"] = np.array([(lambda d: d["n_selected"] / d["n_features_total"])(json.loads((SR / f"{ds}_B_{o}_s{s}_m-desc_b100_lam0.01abs.json").read_text())) for s in range(30)])
        for c, (cond, mk) in enumerate((("S", "o"), ("M", "s"), ("D", "D"))):
            v = fr[cond]; x = k + (c - 1) * 0.25; q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([x, x], [q1, q3], color=COLORS[m], lw=1.1); ax.plot(x, q2, marker=mk, color=COLORS[m], mfc="white" if cond != "S" else COLORS[m], ms=4)
    ax.axhline(0.5, color="#999999", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(["RS", "TPE", "GWO", "WOA", "PSO"])
    ax.set_title(f"Mirror check, {ds}", loc="left")
    if a_ == 0: ax.set_ylabel("Fraction")
    else: ax.set_yticklabels([])
    ax.set_ylim(0.25, 0.75)
fig.legend(handles=[Line2D([], [], marker="o", color="grey", ls="", label="standard"), Line2D([], [], marker="s", mfc="white", color="grey", ls="", label="all mirrored"),
                    Line2D([], [], marker="D", mfc="white", color="grey", ls="", label="descriptor genes mirrored")], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.11)
save(fig, "fig5_feature_selection_combined")

json.dump(AUDIT, open(OUT / "font_audit.json", "w"), indent=1); print(json.dumps(AUDIT, indent=0))
