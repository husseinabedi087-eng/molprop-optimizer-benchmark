"""One function per manuscript figure. Each returns the list of files it wrote.

Every statistic drawn (medians, CIs, p-values, rho, stability, ranks, A12, headroom) is read from
analysis/final/final_stats.json (mirror tests: analysis/final/5_mirror_prespecified_tests.csv, the table in
RESULTS.md section 5; sphere: results/xgb_mh/sanity). Raw per-seed run data are used only to draw
distributions (boxes, jittered points, scatter, paired lines, convergence curves).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter

from data import (AXIS_LABEL, DATASETS, EXPERIMENT_LABEL, EXPERIMENTS, HIGHER_BETTER, METRIC_NAME, SANITY, TASK, Study)
from style import (COLORS, DEFAULT_LINE, DOUBLE_COL, GRID, INK_MUTED, MARKERS, METHODS, OPTIMIZERS, SINGLE_COL,
                   panel_label, save, watermark)

DRAFT_TEXT = "DRAFT: a0 variants"
SIG = "*"


def _finish(fig, study: Study, name: str, out: Path) -> list[Path]:
    if study is not None and study.draft:
        watermark(fig, DRAFT_TEXT)
    return save(fig, name, out)


def _cell(study: Study, ds: str, exp: str):
    df = study.runs
    return df[(df.dataset == ds) & (df.experiment == exp)]


def _values(sub, method: str, column: str = "test_primary") -> np.ndarray:
    return sub[sub.method == method].sort_values("seed")[column].to_numpy(dtype=float)


def _method_handles(include_default: bool = True, marker_size: float = 4, lines: bool = False) -> list[Line2D]:
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="-" if lines else "none",
                      linewidth=0.9, markersize=marker_size, label=m) for m in OPTIMIZERS]
    if include_default:
        handles.insert(0, Line2D([], [], label="Default", **DEFAULT_LINE))
    return handles


def _records(fs: dict, key: str) -> pd.DataFrame:
    return pd.DataFrame(fs[key])


def _fp(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


# ========================================================================================== MAIN FIGURES
def fig2_test_distributions(study: Study, fs: dict, out: Path) -> list[Path]:
    """Test primary metric: box + 30 seeds per optimizer; default dashed; * = Holm-significant vs default."""
    tvd = _records(fs, "tuning_vs_default").query("role == 'primary'").set_index(["dataset", "experiment", "method"])
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 8.6), sharey="row")
    rng = np.random.default_rng(0)
    pos = np.arange(1, len(OPTIMIZERS) + 1)
    for i, ds in enumerate(DATASETS):
        for j, exp in enumerate(EXPERIMENTS):
            ax, sub = axes[i, j], _cell(study, ds, exp)
            data = [_values(sub, m) for m in OPTIMIZERS]
            bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False, patch_artist=True,
                            medianprops=dict(color="black", linewidth=0.9))
            for k, m in enumerate(OPTIMIZERS):
                bp["boxes"][k].set(facecolor=to_rgba(COLORS[m], 0.18), edgecolor=COLORS[m], linewidth=0.8)
                for part in ("whiskers", "caps"):
                    for line in bp[part][2 * k:2 * k + 2]:
                        line.set(color=COLORS[m], linewidth=0.8)
                ax.scatter(pos[k] + rng.uniform(-0.17, 0.17, len(data[k])), data[k], s=6, marker=MARKERS[m],
                           color=COLORS[m], alpha=0.85, linewidths=0, zorder=3)
                if tvd.loc[(ds, exp, m), "p_holm"] < 0.05:
                    ax.annotate(SIG, (pos[k], data[k].max()), xytext=(0, 1), textcoords="offset points",
                                ha="center", va="bottom", fontsize=9, fontweight="bold")
            ax.axhline(tvd.loc[(ds, exp, OPTIMIZERS[0]), "default"], zorder=2, **DEFAULT_LINE)
            ax.set_xticks(pos, OPTIMIZERS if i == len(DATASETS) - 1 else [""] * len(pos))
            ax.set_xlim(0.4, len(pos) + 0.6)
            ax.yaxis.grid(True, **GRID)
            ax.margins(y=0.10)
            if j == 0:
                ax.set_ylabel(f"{ds}\n{AXIS_LABEL[TASK[ds]]}")
            if i == 0:
                ax.text(0.5, 1.10, EXPERIMENT_LABEL[exp], transform=ax.transAxes, ha="center", va="bottom")
            panel_label(ax, 2 * i + j, x=-0.13 if j == 0 else -0.08)
    fig.legend(handles=[Line2D([], [], label="Default XGBoost (deterministic)", **DEFAULT_LINE),
                        Line2D([], [], color="black", marker="$*$", linestyle="none", markersize=6,
                               label="differs from default (one-sample Wilcoxon, Holm over 5, p < 0.05)")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.04, hspace=0.32, wspace=0.08)
    return _finish(fig, study, "fig2_test_distributions", out)


def fig3_forest_vs_default(study: Study, fs: dict, out: Path) -> list[Path]:
    """Median improvement over default with bootstrap 95% CI (final_stats.json); positive = better."""
    tvd = _records(fs, "tuning_vs_default").query("role == 'primary'").set_index(["dataset", "experiment", "method"])
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 7.2), sharex="row", sharey=True)
    y = np.arange(len(OPTIMIZERS))[::-1]
    for i, ds in enumerate(DATASETS):
        task = TASK[ds]
        lim = 1.15 * max(max(abs(tvd.loc[(ds, e, m), "impr_ci_low"]), abs(tvd.loc[(ds, e, m), "impr_ci_high"]))
                         for e in EXPERIMENTS for m in OPTIMIZERS)
        for j, exp in enumerate(EXPERIMENTS):
            ax = axes[i, j]
            ax.axvline(0, color="black", linewidth=0.8, zorder=1)
            ax.xaxis.grid(True, **GRID)
            for k, m in enumerate(OPTIMIZERS):
                r = tvd.loc[(ds, exp, m)]
                ax.hlines(y[k], r.impr_ci_low, r.impr_ci_high, color=COLORS[m], linewidth=1.3, zorder=2)
                ax.plot(r.median_improvement, y[k], marker=MARKERS[m], color=COLORS[m], markersize=4.5, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.4)
                if r.p_holm < 0.05:
                    ax.annotate(SIG, (r.impr_ci_high, y[k]), xytext=(2, -1), textcoords="offset points",
                                ha="left", va="center", fontsize=9, fontweight="bold")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.7, len(OPTIMIZERS) - 0.3)
            ax.set_yticks(y, OPTIMIZERS)
            ax.text(0.99, 1.02, "better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK_MUTED)
            ax.text(0.01, 1.02, f"default {METRIC_NAME[task]} = {tvd.loc[(ds, exp, 'Random'), 'default']:.3f}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color=INK_MUTED)
            if j == 0:
                sign = "method − default" if HIGHER_BETTER[task] else "default − method"
                ax.set_ylabel(f"{ds}\nΔ{METRIC_NAME[task]}\n({sign})")
            if i == 0:
                ax.text(0.5, 1.20, EXPERIMENT_LABEL[exp], transform=ax.transAxes, ha="center", va="bottom")
            panel_label(ax, 2 * i + j, x=-0.20 if j == 0 else -0.06, y=1.10)
    fig.text(0.565, 0.035, "Median improvement over default, bootstrap 95% CI (metric units of each row; > 0 = better)",
             ha="center", va="bottom")
    fig.text(0.99, 0.005, "* one-sample Wilcoxon vs default, Holm over the 5 optimizers, p < 0.05", ha="right",
             va="bottom", fontsize=7, color=INK_MUTED)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.93, bottom=0.09, hspace=0.62, wspace=0.08)
    return _finish(fig, study, "fig3_forest_vs_default", out)


def _tight_limits_with_identity(x: np.ndarray, y: np.ndarray, margin: float = 0.04,
                                min_overlap: float = 0.15) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits hugging the data (margin = fraction of each range), extended only as far as needed so the
    y = x line crosses at least `min_overlap` of the smaller axis span inside the panel."""
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    px, py = margin * (x1 - x0), margin * (y1 - y0)
    x0, x1, y0, y1 = x0 - px, x1 + px, y0 - py, y1 + py
    need = min_overlap * min(x1 - x0, y1 - y0)
    deficit = need - (min(x1, y1) - max(x0, y0))  # length of y = x inside the box, projected on one axis
    if deficit > 0:
        if x0 > y0:          # x lower bound is the binding one: lower it
            x0 -= deficit
        else:
            y0 -= deficit
    return (x0, x1), (y0, y1)


def fig4_validation_reliability(study: Study, fs: dict, out: Path) -> list[Path]:
    """(a-e) best validation vs test per run; (f) Spearman rho with bootstrap 95% CI from final_stats.json."""
    vr = _records(fs, "validation_reliability").set_index(["dataset", "experiment", "method"])
    fig = plt.figure(figsize=(DOUBLE_COL, 7.4))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 1.25], left=0.08, right=0.98, top=0.93, bottom=0.07,
                  hspace=0.45, wspace=0.55)
    slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1], gs[2, 0]]
    for i, (ds, slot) in enumerate(zip(DATASETS, slots)):
        ax, task = fig.add_subplot(slot), TASK[ds]
        vals, tests = [], []
        for exp in EXPERIMENTS:
            sub = _cell(study, ds, exp)
            for m in OPTIMIZERS:
                v, t = _values(sub, m, "val_primary"), _values(sub, m)
                ax.scatter(v, t, s=9, marker=MARKERS[m], facecolors=COLORS[m] if exp == "A" else "none",
                           edgecolors=COLORS[m], linewidths=0.6, alpha=0.75, zorder=3)
                vals.append(v)
                tests.append(t)
        dv, dt = vr.loc[(ds, "A", "Default"), "median_val"], vr.loc[(ds, "A", "Default"), "median_test"]
        ax.scatter([dv], [dt], marker="*", s=70, color="black", zorder=4, linewidths=0)
        (x0, x1), (y0, y1) = _tight_limits_with_identity(np.r_[np.concatenate(vals), dv], np.r_[np.concatenate(tests), dt])
        ax.plot([min(x0, y0), max(x1, y1)], [min(x0, y0), max(x1, y1)], color="#BDBDBD", linewidth=0.6, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_box_aspect(1)
        ax.set_xlabel(f"Best validation {METRIC_NAME[task]}")
        ax.set_ylabel(f"Test {METRIC_NAME[task]}")
        ax.text(0.0, 1.03, ds, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold")
        panel_label(ax, i, x=-0.36, y=1.03)
    leg = fig.add_subplot(gs[2, 1])
    leg.axis("off")
    leg.legend(handles=_method_handles(include_default=False) + [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="Experiment A (filled)"),
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, markerfacecolor="none",
               label="Experiment B (open)"),
        Line2D([], [], color="black", marker="*", linestyle="none", markersize=7, label="Default XGBoost"),
        Line2D([], [], color="#BDBDBD", linewidth=0.6, label="y = x")], loc="center left", frameon=False)

    ax = fig.add_subplot(gs[:, 2])
    ytick, ylab, gap = [], [], len(OPTIMIZERS) + 1.2
    for di, ds in enumerate(DATASETS):
        base = -di * gap
        for mi, m in enumerate(OPTIMIZERS):
            for exp, dy, face in (("A", 0.17, COLORS[m]), ("B", -0.17, "white")):
                r = vr.loc[(ds, exp, m)]
                yy = base - mi + dy
                ax.hlines(yy, r.rho_ci_low, r.rho_ci_high, color=COLORS[m], linewidth=0.9)
                ax.plot(r.rho, yy, marker=MARKERS[m], markersize=3.8, color=COLORS[m], markerfacecolor=face,
                        markeredgewidth=0.7)
        ytick.append(base - (len(OPTIMIZERS) - 1) / 2)
        ylab.append(ds)
        if di:
            ax.axhline(base + 1.1, color="#E6E6E6", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(ytick, ylab)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ (best val. vs test)\nover 30 seeds, bootstrap 95% CI")
    ax.xaxis.grid(True, **GRID)
    ax.text(0.02, 1.005, "filled = Exp. A, open = Exp. B", transform=ax.transAxes, fontsize=6.5, color=INK_MUTED,
            va="bottom")
    panel_label(ax, 5, x=-0.30, y=1.02)
    return _finish(fig, study, "fig4_validation_reliability", out)


def fig5_feature_selection(study: Study, fs: dict, pairs: pd.DataFrame, mtests: pd.DataFrame, out: Path) -> list[Path]:
    """(a) selected fraction, (b) Nogueira stability, (c) mirrored-encoding check."""
    fsel = _records(fs, "feature_selection").set_index(["dataset", "method"])
    fig = plt.figure(figsize=(DOUBLE_COL, 7.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 0.9, 1.25], left=0.10, right=0.99, top=0.95, bottom=0.10,
                  hspace=0.42, wspace=0.12)
    ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    offsets = np.linspace(-0.32, 0.32, len(OPTIMIZERS))
    rng = np.random.default_rng(1)
    for g, ds in enumerate(DATASETS):
        sub = _cell(study, ds, "B")
        for k, m in enumerate(OPTIMIZERS):
            runs = sub[sub.method == m]
            frac = runs.n_selected.to_numpy() / runs.n_features_total.to_numpy()
            xpos = g + offsets[k]
            bp = ax1.boxplot([frac], positions=[xpos], widths=0.12, showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", linewidth=0.8))
            bp["boxes"][0].set(facecolor=to_rgba(COLORS[m], 0.18), edgecolor=COLORS[m], linewidth=0.7)
            for line in bp["whiskers"] + bp["caps"]:
                line.set(color=COLORS[m], linewidth=0.7)
            ax1.scatter(xpos + rng.uniform(-0.04, 0.04, len(frac)), frac, s=4, marker=MARKERS[m], color=COLORS[m],
                        alpha=0.8, linewidths=0, zorder=3)
            r = fsel.loc[(ds, m)]
            ax2.errorbar(xpos, r.nogueira, yerr=[[r.nogueira - r.nogueira_ci_low], [r.nogueira_ci_high - r.nogueira]],
                         fmt=MARKERS[m], color=COLORS[m], markersize=4, elinewidth=1.0, capsize=0,
                         markeredgecolor="white", markeredgewidth=0.4)
    ax1.axhline(0.5, color="#BDBDBD", linewidth=0.6, zorder=1)
    ax1.text(0.005, 0.02, "grey line: 0.5 (uniform random genes)", transform=ax1.transAxes, fontsize=6.5,
             color=INK_MUTED, va="bottom")
    ax1.set_ylabel("Selected features\n(fraction of F), Exp. B")
    ax1.yaxis.grid(True, **GRID)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.text(0.005, 0.97, "black line: Φ = 0 = random selection", transform=ax2.transAxes, fontsize=6.5,
             color=INK_MUTED, ha="left", va="top")
    ax2.set_ylabel("Nogueira stability Φ\n(95% CI)")
    ax2.yaxis.grid(True, **GRID)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(DATASETS)), [f"{ds} (F = {int(fsel.loc[(ds, 'Random'), 'F'])})" for ds in DATASETS])
        ax.set_xlim(-0.55, len(DATASETS) - 0.45)
    ax1.set_xticklabels([])
    panel_label(ax1, 0, x=-0.075)
    panel_label(ax2, 1, x=-0.075)
    fig.legend(handles=_method_handles(include_default=False), loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.0))

    # (c) mirror check: paired normal vs mirrored per seed
    for c, ds in enumerate(("BBBP", "FreeSolv")):
        ax = fig.add_subplot(gs[2, c])
        sub = pairs[pairs.dataset == ds]
        for k, m in enumerate(OPTIMIZERS):
            p = sub[sub.method == m].sort_values("seed")
            emph = m == "GWO"
            xa, xb = k - 0.18, k + 0.18
            for fn, fm in zip(p.frac_normal, p.frac_mirrored):
                ax.plot([xa, xb], [fn, fm], color=COLORS[m] if emph else "#C8C8C8", linewidth=0.5 if emph else 0.35,
                        alpha=0.7 if emph else 0.8, zorder=1)
            ax.scatter(np.r_[[xa] * len(p), [xb] * len(p)], np.r_[p.frac_normal, p.frac_mirrored], s=5 if emph else 3,
                       marker=MARKERS[m], color=COLORS[m], alpha=0.9 if emph else 0.45, linewidths=0, zorder=2)
            t = mtests[(mtests.dataset == ds) & (mtests.method == m) & (mtests.outcome == "fraction")].iloc[0]
            for xx, med in ((xa, t.median_normal), (xb, t.median_mirrored)):
                ax.hlines(med, xx - 0.09, xx + 0.09, color="black", linewidth=1.8 if emph else 1.1, zorder=3)
            ax.text(k, 1.0, "<.001" if t.p_holm < 1e-3 else f"{t.p_holm:.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6.5, fontweight="bold" if t.p_holm < 0.05 else "normal",
                    color="black" if t.p_holm < 0.05 else INK_MUTED)
        ax.axhline(0.5, color="#BDBDBD", linewidth=0.6, zorder=0)
        ax.set_xticks([k + d for k in range(len(OPTIMIZERS)) for d in (-0.18, 0.18)], ["N", "M"] * len(OPTIMIZERS),
                      fontsize=6.5)
        ax.tick_params(axis="x", length=2, pad=1)
        for k, m in enumerate(OPTIMIZERS):
            ax.text(k, -0.12, "Random\n(control)" if m == "Random" else m, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, fontweight="bold" if m == "GWO" else "normal")
        ax.text(-0.60, 1.0, "Holm p", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=6.5,
                color=INK_MUTED)
        ax.set_xlim(-0.55, len(OPTIMIZERS) - 0.45)
        ax.set_ylim(0.0, 1.13)  # headroom so the p-value row sits above the 1.0 tick
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.grid(True, **GRID)
        ax.text(0.01, 0.02, f"{ds}, Exp. B", transform=ax.transAxes, fontsize=7, va="bottom")
        if c == 0:
            ax.set_ylabel("Selected fraction\nN = normal, M = mirrored")
            panel_label(ax, 2, x=-0.17)
        else:
            ax.set_yticklabels([])
    fig.text(0.99, 0.005, "c: lines = seeds (paired); bars = medians; paired Wilcoxon, Holm over 5 optimizers "
             "(pre-registered)", ha="right", va="bottom", fontsize=6.5, color=INK_MUTED)
    return _finish(fig, study, "fig5_feature_selection", out)


def figS4_headroom_exploratory(fs: dict, out: Path) -> list[Path]:
    """EXPLORATORY: validation headroom (%) vs test gain (%) over the 10 dataset x experiment cells."""
    h = fs["headroom_EXPLORATORY"]
    cells = pd.DataFrame(h["cells"])
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 3.3))
    lo = min(0, cells.test_improvement_pct.min()) - 2
    hi = cells.val_headroom_pct.max() + 3
    ax.plot([lo, hi], [lo, hi], color="#BDBDBD", linewidth=0.6, zorder=1)
    ax.axhline(0, color="black", linewidth=0.6, zorder=1)
    # Label positions (data units) chosen to clear the cluster near the origin; thin leader lines to each point.
    label_at = {("ESOL", "A"): (19.5, -5.0), ("ESOL", "B"): (19.5, -1.2), ("FreeSolv", "A"): (18.0, 13.5),
                ("FreeSolv", "B"): (24.0, 16.5), ("Lipophilicity", "A"): (9.0, 7.0), ("Lipophilicity", "B"): (9.0, 5.0),
                ("BACE", "B"): (9.0, 2.6), ("BACE", "A"): (9.0, 0.6), ("BBBP", "B"): (6.0, -2.2),
                ("BBBP", "A"): (6.0, -4.3)}  # BBBP labels below/right of y = x so leaders never cross it
    for r in cells.itertuples():
        mk = "o" if r.experiment == "A" else "s"
        ax.scatter(r.val_headroom_pct, r.test_improvement_pct, marker=mk, s=18, facecolor="white" if r.experiment == "B"
                   else "black", edgecolor="black", linewidths=0.7, zorder=3)
        tx, ty = label_at[(r.dataset, r.experiment)]
        ax.annotate(f"{r.dataset} {r.experiment}", (r.val_headroom_pct, r.test_improvement_pct), xytext=(tx, ty),
                    textcoords="data", fontsize=6.5, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#9E9E9E", linewidth=0.4, shrinkA=0, shrinkB=2))
    ax.set_xlim(-6, hi)
    ax.set_ylim(lo - 1, 20)
    ax.set_xlabel("Validation headroom over default (%)\n(median best val. of the 5 optimizers)")
    ax.set_ylabel("Test gain over default (%)\n(median of the 5 optimizers)")
    ax.legend(handles=[Line2D([], [], marker="o", color="black", linestyle="none", markersize=4, label="Exp. A"),
                       Line2D([], [], marker="s", color="black", markerfacecolor="white", linestyle="none",
                              markersize=4, label="Exp. B"),
                       Line2D([], [], color="#BDBDBD", linewidth=0.6, label="y = x")], loc="upper left")
    ax.text(0.03, 0.72, f"EXPLORATORY\n(not pre-specified)\nSpearman ρ = {h['spearman_rho_pct']:+.2f}\n"
            f"p = {h['p_pct']:.2f}, n = {h['n_cells']} cells", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, color=INK_MUTED)
    ax.yaxis.grid(True, **GRID)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.97, bottom=0.2)
    return save(fig, "figS4_headroom_EXPLORATORY", out)


# ========================================================================================== SUPPLEMENTARY
def figS6_convergence(study: Study, curves: dict, fs: dict, out: Path) -> list[Path]:
    """Median best-so-far validation fitness vs call index, IQR band (curves: raw call logs)."""
    vr = _records(fs, "validation_reliability").set_index(["dataset", "experiment", "method"])
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 8.8), sharey="row", sharex=True)
    x = np.arange(1, 101)
    marks = [19, 39, 59, 79, 99]
    for i, ds in enumerate(DATASETS):
        task = TASK[ds]
        dval = vr.loc[(ds, "A", "Default"), "median_val"]
        base = (1 - dval) if task == "classification" else dval  # default fitness (raw metric)
        tops, bottoms = [], []
        for j, exp in enumerate(EXPERIMENTS):
            ax = axes[i, j]
            for m in OPTIMIZERS:
                c = curves[(ds, exp, m)]
                med, q1, q3 = np.median(c, 0), np.percentile(c, 25, 0), np.percentile(c, 75, 0)
                ax.fill_between(x, q1, q3, color=COLORS[m], alpha=0.13, linewidth=0)
                ax.plot(x, med, color=COLORS[m], linewidth=0.9, marker=MARKERS[m], markevery=marks, markersize=3)
                tops.append(q3[9])
                bottoms.append(q1[-1])
            default_fitness = base + (0.01 if exp == "B" else 0.0)  # Exp B: all features -> penalty 0.01
            ax.axhline(default_fitness, **DEFAULT_LINE)
            tops.append(default_fitness)
            ax.axvline(10.5, color="#BDBDBD", linewidth=0.5)
            ax.yaxis.grid(True, **GRID)
            ax.set_xlim(1, 100)
            if j == 0:
                ax.set_ylabel("RMSE" if task == "regression" else "1 − AUC")
                ax.text(0.0, 1.04, ds, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")
            if i == 0:
                ax.text(0.5, 1.22, EXPERIMENT_LABEL[exp] + ("\n(fitness incl. feature-count penalty)" if exp == "B" else "\n"),
                        transform=ax.transAxes, ha="center", va="bottom")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Fitness call")
            panel_label(ax, 2 * i + j, x=-0.13 if j == 0 else -0.06, y=1.04)
        span = max(tops) - min(bottoms)
        axes[i, 0].set_ylim(min(bottoms) - 0.05 * span, max(tops) + 0.08 * span)
    handles = _method_handles(marker_size=3, lines=True) + [
        Line2D([], [], color="#BDBDBD", linewidth=0.8, label="end of shared initial design (call 10)")]
    fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(0.54, 1.0), fontsize=6.5,
               columnspacing=1.0, handletextpad=0.4)
    fig.text(0.01, 0.5, "Median best-so-far validation fitness (IQR band)", rotation=90, va="center", ha="left")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.05, hspace=0.42, wspace=0.08)
    return save(fig, "figS6_convergence", out)


def _draw_cd(ax, names: list[str], ranks: np.ndarray, cd: float, header: str) -> None:
    k = len(names)
    order = np.argsort(ranks)
    names, ranks = [names[i] for i in order], ranks[order]
    axis_y = 0.78
    ax.set_xlim(-1.7, k + 2.7)
    ax.set_ylim(0, 1.34)
    ax.axis("off")
    ax.hlines(axis_y, 1, k, color="black", linewidth=0.8)
    for t in range(1, k + 1):
        ax.vlines(t, axis_y, axis_y + 0.035, color="black", linewidth=0.8)
        ax.text(t, axis_y + 0.05, str(t), ha="center", va="bottom", fontsize=7)
    ax.hlines(1.0, 1, 1 + cd, color="black", linewidth=1.2)
    for xx in (1, 1 + cd):
        ax.vlines(xx, 0.975, 1.025, color="black", linewidth=0.8)
    ax.text(1 + cd / 2, 1.04, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    half = int(np.ceil(k / 2))
    for idx, (name, r) in enumerate(zip(names, ranks)):
        left = idx < half
        level = idx if left else k - 1 - idx
        yy = axis_y - 0.16 - 0.11 * level
        xe = 0.6 if left else k + 0.4
        color = COLORS.get(name, "black")
        ax.plot([r, r, xe], [axis_y, yy, yy], color=color, linewidth=0.8)
        ax.text(xe - 0.08 if left else xe + 0.08, yy, f"{name} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7)
    cliques, last_j = [], -1
    for i in range(k):
        j = max(jj for jj in range(i, k) if ranks[jj] - ranks[i] < cd)
        if j > i and j > last_j:
            cliques.append((i, j))
            last_j = j
    for n, (i, j) in enumerate(cliques):
        yy = axis_y - 0.06 - 0.035 * n
        ax.hlines(yy, ranks[i] - 0.04, ranks[j] + 0.04, color="black", linewidth=2.2)
    ax.text(-1.15, 1.34, header, ha="left", va="top", fontsize=7, color=INK_MUTED)


def figS3_cd_optimizers(fs: dict, out: Path) -> list[Path]:
    """CD diagrams across datasets, optimizers only, from final_stats.json cd_ranks (low-powered, N = 5)."""
    cd = _records(fs, "cd_ranks")
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.6))
    for idx, exp in enumerate(EXPERIMENTS):
        g = cd[cd.experiment == exp].set_index("method").loc[list(OPTIMIZERS)]
        _draw_cd(axes[idx], list(OPTIMIZERS), g.mean_rank.to_numpy(), float(g.CD.iloc[0]),
                 f"Experiment {exp}: optimizers, mean rank over N = {int(g.N_datasets.iloc[0])} datasets\n"
                 f"LOW-POWERED; Nemenyi CD at α = 0.05; Friedman p = {g.friedman_p_across_datasets.iloc[0]:.2f}")
        panel_label(axes[idx], idx, x=0.0, y=0.93)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02, hspace=0.15)
    return save(fig, "figS3_cd_optimizers", out)


def figS2_a12_heatmap(fs: dict, out: Path) -> list[Path]:
    """Lower triangle: A12 of row vs column optimizer (primary metric); * = Holm-significant pair."""
    pw = _records(fs, "pairwise").query("role == 'primary'")
    cmap = LinearSegmentedColormap.from_list("div", ["#B2182B", "#F2F2F2", "#2166AC"])
    n = len(OPTIMIZERS)
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(DOUBLE_COL, 3.8))
    im = None
    for i, exp in enumerate(EXPERIMENTS):
        for j, ds in enumerate(DATASETS):
            ax = axes[i, j]
            g = pw[(pw.dataset == ds) & (pw.experiment == exp)].set_index("pair")
            mat, sig = np.full((n, n), np.nan), np.zeros((n, n), bool)
            for r in range(n):
                for c in range(r):
                    rec = g.loc[f"{OPTIMIZERS[c]} vs {OPTIMIZERS[r]}"]      # stored as (column vs row)
                    mat[r, c] = 1.0 - rec.A12                              # A12(row vs column)
                    sig[r, c] = rec.p_holm < 0.05
            im = ax.imshow(mat, cmap=cmap, vmin=0.2, vmax=0.8)
            for r in range(n):
                for c in range(r):
                    v = mat[r, c]
                    ax.text(c, r, f"{v:.2f}".lstrip("0") + ("*" if sig[r, c] else ""), ha="center", va="center",
                            fontsize=6.5, fontweight="bold" if sig[r, c] else "normal",
                            color="white" if abs(v - 0.5) > 0.22 else "black")
                    if sig[r, c]:
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.0))
            ax.set_xticks(range(n - 1), OPTIMIZERS[:-1] if i == 1 else [""] * (n - 1), rotation=45, ha="right",
                          rotation_mode="anchor")
            ax.set_yticks(range(1, n), OPTIMIZERS[1:] if j == 0 else [""] * (n - 1))
            ax.set_xlim(-0.5, n - 1.5)
            ax.set_ylim(n - 0.5, 0.5)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.text(0.5, 1.06, ds, transform=ax.transAxes, ha="center", va="bottom")
            if j == 0:
                ax.set_ylabel(f"Experiment {exp}")
            panel_label(ax, i * len(DATASETS) + j, x=-0.30 if j == 0 else -0.08, y=1.02)
    cax = fig.add_axes([0.925, 0.2, 0.012, 0.6])
    cb = fig.colorbar(im, cax=cax, ticks=[0.2, 0.35, 0.5, 0.65, 0.8])
    cb.outline.set_visible(False)
    cb.set_label("A12 (row better than column →)")
    fig.text(0.01, 0.01, "* boxed = pairwise Wilcoxon (pratt), Holm over 10 pairs, p < 0.05", fontsize=6.5, color=INK_MUTED)
    fig.subplots_adjust(left=0.10, right=0.91, top=0.92, bottom=0.16, wspace=0.12, hspace=0.18)
    return save(fig, "figS2_a12_heatmap", out)


def figS1_sphere(out: Path) -> list[Path]:
    """Sphere sanity results (results/xgb_mh/sanity): gwo_ref / woa_ref primary, a0 variants faded."""
    st = pd.read_csv(SANITY / "sphere_tests.csv").set_index("optimizer")
    cb = pd.read_csv(SANITY / "sphere_corner_bias.csv").set_index("optimizer")
    order = [("random", "Random", False), ("tpe", "TPE", False), ("gwo_ref", "GWO", False), ("gwo", "GWO a0", True),
             ("woa_ref", "WOA", False), ("woa", "WOA a0", True), ("pso", "PSO", False)]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.8))
    specs = [(0, 10, "centered", "shifted", "optimum at 0.5 (filled) vs shifted, fixed random point (open)"),
             (1, 200, "centered", "shifted", None),
             (2, 10, "0.2", "0.8", "optimum at 0.2 in every dim. (filled) vs 0.8 (open)"),
             (3, 200, "0.2", "0.8", None)]
    for idx, d, a, b, note in specs:
        ax = axes.ravel()[idx]
        for k, (key, label, faded) in enumerate(order):
            color = COLORS[label.split()[0]]
            alpha = 0.35 if faded else 1.0
            mk = MARKERS[label.split()[0]]
            for dx, which, face in ((-0.14, a, color), (0.14, b, "white")):
                if a == "centered":
                    med = st.loc[key, f"d{d}_{which}_median"]
                    lo, hi = map(float, st.loc[key, f"d{d}_{which}_iqr"].split("-"))
                    ax.vlines(k + dx, lo, hi, color=color, alpha=alpha, linewidth=0.9)
                else:
                    med = cb.loc[key, f"d{d}_opt_at_{which}"]
                ax.plot(k + dx, med, marker=mk, color=color, markerfacecolor=face, alpha=alpha, markersize=4.5,
                        markeredgewidth=0.8)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(order)), [o[1] for o in order], rotation=30, ha="right", rotation_mode="anchor")
        ax.yaxis.grid(True, which="major", **GRID)
        ax.set_ylabel(f"Median best fitness, d = {d}")
        if note:
            ax.text(0.0, 1.06, note, transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
        panel_label(ax, idx, x=-0.16, y=1.04)
    fig.text(0.99, 0.005, "budget 100, 30 seeds per point; bars in a, b = IQR; faded = a0 variants (sensitivity only); "
             "sphere f(x) = Σ(x − c)², minimised; lower is better", ha="right", va="bottom", fontsize=6.5, color=INK_MUTED)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.13, hspace=0.62, wspace=0.25)
    return save(fig, "figS1_sphere", out)


def figS5_sensitivity(study: Study, a0: pd.DataFrame, fs: dict, out: Path) -> list[Path]:
    """a0 - ref paired differences (primary metric; > 0 = a0 better). Medians and Holm p from final_stats.json."""
    sens = _records(fs, "sensitivity_paired").set_index(["method", "dataset", "experiment"])
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 6.6), sharey=True)
    rng = np.random.default_rng(2)
    for i, ds in enumerate(DATASETS):
        hb = HIGHER_BETTER[TASK[ds]]
        for j, m in enumerate(("GWO", "WOA")):
            ax = axes[i, j]
            for k, exp in enumerate(EXPERIMENTS):
                ref = _values(_cell(study, ds, exp), m)
                z = a0[(a0.dataset == ds) & (a0.experiment == exp) & (a0.method == m)].sort_values("seed").test_primary.to_numpy()
                diff = (z - ref) if hb else (ref - z)  # a0 - ref in goodness units: > 0 = a0 better
                yk = 1 - k
                ax.scatter(diff, yk + rng.uniform(-0.12, 0.12, len(diff)), s=4, color=COLORS[m], alpha=0.35, linewidths=0)
                r = sens.loc[(m, ds, exp)]
                med = -r.median_paired_diff_ref_better
                ax.plot(med, yk, marker="|", markersize=12, markeredgewidth=2.0, color="black", zorder=3)
                ax.text(1.0, yk, f"  {_fp(r.p_holm).replace('p ', 'Holm p ')}", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=6.5, color=INK_MUTED)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_yticks([1, 0], ["Exp. A", "Exp. B"])
            ax.set_ylim(-0.6, 1.6)
            ax.xaxis.grid(True, **GRID)
            lim = np.nanmax(np.abs(ax.get_xlim()))
            ax.set_xlim(-lim, lim)
            if j == 0:
                ax.text(-0.22, 0.5, f"{ds}\nΔ{METRIC_NAME[TASK[ds]]}", transform=ax.transAxes, ha="center",
                        va="center", rotation=90)
            if i == 0:
                ax.text(0.5, 1.12, f"{m}: a0 − reference schedule", transform=ax.transAxes, ha="center", va="bottom")
            ax.text(0.99, 1.01, "a0 better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK_MUTED)
            panel_label(ax, 2 * i + j, x=-0.30 if j == 0 else -0.10, y=1.02)
    fig.text(0.99, 0.005, "dots = seeds (paired); bar = median paired difference; Holm over the 10 dataset x "
             "experiment cells per method", ha="right", va="bottom", fontsize=6.5, color=INK_MUTED)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.92, bottom=0.06, hspace=0.55, wspace=0.40)
    return save(fig, "figS5_sensitivity", out)
