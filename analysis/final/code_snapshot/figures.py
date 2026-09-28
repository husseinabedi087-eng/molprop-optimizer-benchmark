"""One function per manuscript figure. Each returns the list of files it wrote."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare, spearmanr

from data import (AXIS_LABEL, DATASETS, EXPERIMENT_LABEL, EXPERIMENTS, HIGHER_BETTER, METRIC_NAME, TASK, Study,
                  feature_mask)
from stats import a12, average_ranks, bootstrap_median_ci, nemenyi_cd, nogueira_stability
from style import (COLORS, DEFAULT_LINE, DOUBLE_COL, GRID, INK_MUTED, MARKERS, METHODS, OPTIMIZERS, SINGLE_COL,
                   panel_label, save, watermark)

DRAFT_TEXT = "DRAFT: a0 variants"


def _finish(fig, study: Study, name: str, out: Path) -> list[Path]:
    if study.draft:
        watermark(fig, DRAFT_TEXT)
    return save(fig, name, out)


def _cell(study: Study, ds: str, exp: str):
    df = study.runs
    return df[(df.dataset == ds) & (df.experiment == exp)]


def _values(sub, method: str, column: str = "test_primary") -> np.ndarray:
    return sub[sub.method == method].sort_values("seed")[column].to_numpy(dtype=float)


def _goodness(ds: str, x: np.ndarray) -> np.ndarray:
    return x if HIGHER_BETTER[TASK[ds]] else -x


def _method_handles(include_default: bool = True, marker_size: float = 4, lines: bool = False) -> list[Line2D]:
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], linestyle="-" if lines else "none",
                      linewidth=0.9, markersize=marker_size, label=m) for m in OPTIMIZERS]
    if include_default:
        handles.insert(0, Line2D([], [], label="Default", **DEFAULT_LINE))
    return handles


# ----------------------------------------------------------------------------------------------------------
def fig2_test_distributions(study: Study, out: Path) -> list[Path]:
    """Test primary metric per method: box plot + all 30 seeds; default as dashed line."""
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
            ax.axhline(_values(sub, "Default")[0], zorder=2, **DEFAULT_LINE)
            ax.set_xticks(pos, OPTIMIZERS if i == len(DATASETS) - 1 else [""] * len(pos))
            ax.set_xlim(0.4, len(pos) + 0.6)
            ax.yaxis.grid(True, **GRID)
            if j == 0:
                ax.set_ylabel(f"{ds}\n{AXIS_LABEL[TASK[ds]]}")
            if i == 0:
                ax.text(0.5, 1.10, EXPERIMENT_LABEL[exp], transform=ax.transAxes, ha="center", va="bottom")
            panel_label(ax, 2 * i + j, x=-0.13 if j == 0 else -0.08)
    fig.legend(handles=[Line2D([], [], label="Default XGBoost (deterministic)", **DEFAULT_LINE)],
               loc="upper right", bbox_to_anchor=(0.98, 1.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.04, hspace=0.32, wspace=0.14)
    return _finish(fig, study, "fig2_test_distributions", out)


# ----------------------------------------------------------------------------------------------------------
def fig3_forest_vs_default(study: Study, out: Path) -> list[Path]:
    """Median improvement over the default with bootstrap 95% CI; positive = better than default."""
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 7.4), sharex="row", sharey=True)
    y = np.arange(len(OPTIMIZERS))[::-1]
    for i, ds in enumerate(DATASETS):
        task, stats_row = TASK[ds], []
        for exp in EXPERIMENTS:
            sub = _cell(study, ds, exp)
            d0 = _values(sub, "Default")[0]
            stats_row.append((d0, [bootstrap_median_ci(_goodness(ds, _values(sub, m)) - _goodness(ds, np.array(d0)))
                                   for m in OPTIMIZERS]))
        lim = 1.12 * max(max(abs(lo), abs(hi)) for _, s in stats_row for _, lo, hi in s)
        for j, (exp, (d0, s)) in enumerate(zip(EXPERIMENTS, stats_row)):
            ax = axes[i, j]
            ax.axvline(0, color="black", linewidth=0.8, zorder=1)
            ax.xaxis.grid(True, **GRID)
            for k, (m, (med, lo, hi)) in enumerate(zip(OPTIMIZERS, s)):
                ax.hlines(y[k], lo, hi, color=COLORS[m], linewidth=1.3, zorder=2)
                ax.plot(med, y[k], marker=MARKERS[m], color=COLORS[m], markersize=4.5, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-0.7, len(OPTIMIZERS) - 0.3)
            ax.set_yticks(y, OPTIMIZERS)
            sign = "method − default" if HIGHER_BETTER[task] else "default − method"
            ax.set_xlabel(f"Δ{METRIC_NAME[task]} vs default ({sign})", labelpad=1)
            ax.text(0.99, 1.02, "better →", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK_MUTED)
            ax.text(0.01, 1.02, f"default {METRIC_NAME[task]} = {d0:.3f}", transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=7, color=INK_MUTED)
            if j == 0:
                ax.set_ylabel(ds)
            if i == 0:
                ax.text(0.5, 1.20, EXPERIMENT_LABEL[exp], transform=ax.transAxes, ha="center", va="bottom")
            panel_label(ax, 2 * i + j, x=-0.17 if j == 0 else -0.08, y=1.10)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.06, hspace=0.95, wspace=0.10)
    return _finish(fig, study, "fig3_forest_vs_default", out)


# ----------------------------------------------------------------------------------------------------------
def fig4_val_vs_test(study: Study, out: Path) -> list[Path]:
    """Best validation score vs test score per run; Spearman rho over all optimizer runs per experiment."""
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 4.9))
    flat = axes.ravel()
    for i, ds in enumerate(DATASETS):
        ax, task = flat[i], TASK[ds]
        lo, hi = np.inf, -np.inf
        rhos = []
        for exp in EXPERIMENTS:
            sub = _cell(study, ds, exp)
            opt = sub[sub.method != "Default"]
            rhos.append(spearmanr(opt.val_primary, opt.test_primary)[0])
            for m in OPTIMIZERS:
                v, t = _values(sub, m, "val_primary"), _values(sub, m)
                face = COLORS[m] if exp == "A" else "none"
                ax.scatter(v, t, s=9, marker=MARKERS[m], facecolors=face, edgecolors=COLORS[m], linewidths=0.6,
                           alpha=0.75, zorder=3)
                lo, hi = min(lo, v.min(), t.min()), max(hi, v.max(), t.max())
        dv, dt = _values(_cell(study, ds, "A"), "Default", "val_primary")[0], _values(_cell(study, ds, "A"), "Default")[0]
        ax.scatter([dv], [dt], marker="*", s=70, color="black", zorder=4, linewidths=0)
        lo, hi = min(lo, dv, dt), max(hi, dv, dt)
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#BDBDBD", linewidth=0.6, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Best validation {METRIC_NAME[task]}")
        ax.set_ylabel(f"Test {METRIC_NAME[task]}")
        corner = (0.03, 0.97, "left", "top") if HIGHER_BETTER[task] else (0.97, 0.03, "right", "bottom")
        ax.text(corner[0], corner[1], f"{ds}\nρ(A) = {rhos[0]:+.2f}\nρ(B) = {rhos[1]:+.2f}", transform=ax.transAxes,
                ha=corner[2], va=corner[3], fontsize=7, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
        panel_label(ax, i, x=-0.30)
    leg = flat[-1]
    leg.axis("off")
    handles = _method_handles(include_default=False) + [
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, label="Experiment A (filled)"),
        Line2D([], [], color="black", marker="o", linestyle="none", markersize=4, markerfacecolor="none",
               label="Experiment B (open)"),
        Line2D([], [], color="black", marker="*", linestyle="none", markersize=7, label="Default XGBoost"),
        Line2D([], [], color="#BDBDBD", linewidth=0.6, label="y = x (no validation-test gap)")]
    leg.legend(handles=handles, loc="center left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.09, hspace=0.42, wspace=0.45)
    return _finish(fig, study, "fig4_val_vs_test", out)


# ----------------------------------------------------------------------------------------------------------
def fig5_convergence(study: Study, curves: dict, out: Path) -> list[Path]:
    """Median best-so-far validation fitness vs call index, IQR band."""
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(DOUBLE_COL, 8.6), sharey="row", sharex=True)
    x = np.arange(1, 101)
    marks = [19, 39, 59, 79, 99]
    for i, ds in enumerate(DATASETS):
        task = TASK[ds]
        base = _values(_cell(study, ds, "A"), "Default", "best_val_raw_metric")[0]
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
            default_fitness = base + (0.01 if exp == "B" else 0.0)  # Exp B: all features -> penalty 0.01 * F/F
            ax.axhline(default_fitness, **DEFAULT_LINE)
            tops.append(default_fitness)
            ax.axvline(10.5, color="#BDBDBD", linewidth=0.5)
            ax.yaxis.grid(True, **GRID)
            ax.set_xlim(1, 100)
            if i == 0 and j == 0:
                ax.text(11.5, 0.04, "end of shared initial design", transform=ax.get_xaxis_transform(), fontsize=6.5,
                        color=INK_MUTED, va="bottom")
            if j == 0:
                fit = "val. RMSE" if task == "regression" else "val. 1 − AUC"
                ax.set_ylabel(f"{ds}\n{fit}")
            if i == 0:
                label = EXPERIMENT_LABEL[exp] + ("\nmedian best-so-far fitness (incl. size penalty), IQR band"
                                                 if exp == "B" else "\nmedian best-so-far fitness, IQR band")
                ax.text(0.5, 1.10, label, transform=ax.transAxes, ha="center", va="bottom")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Fitness call")
            panel_label(ax, 2 * i + j, x=-0.13 if j == 0 else -0.08)
        span = max(tops) - min(bottoms)
        axes[i, 0].set_ylim(min(bottoms) - 0.05 * span, max(tops) + 0.05 * span)
    fig.legend(handles=_method_handles(marker_size=3, lines=True), loc="upper center", ncol=6, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.915, bottom=0.05, hspace=0.30, wspace=0.10)
    return _finish(fig, study, "fig5_convergence", out)


# ----------------------------------------------------------------------------------------------------------
def fig6_feature_counts(study: Study, out: Path) -> tuple[list[Path], list[dict]]:
    """Experiment B: fraction of features selected, and Nogueira stability with 95% CI."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL, 4.8), sharex=True)
    offsets = np.linspace(-0.32, 0.32, len(OPTIMIZERS))
    rng = np.random.default_rng(1)
    rows = []
    for g, ds in enumerate(DATASETS):
        sub = _cell(study, ds, "B")
        F = int(sub.n_features_total.iloc[0])
        for k, m in enumerate(OPTIMIZERS):
            runs = sub[sub.method == m].sort_values("seed")
            frac = runs.n_selected.to_numpy() / F
            xpos = g + offsets[k]
            bp = ax1.boxplot([frac], positions=[xpos], widths=0.12, showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", linewidth=0.8))
            bp["boxes"][0].set(facecolor=to_rgba(COLORS[m], 0.18), edgecolor=COLORS[m], linewidth=0.7)
            for line in bp["whiskers"] + bp["caps"]:
                line.set(color=COLORS[m], linewidth=0.7)
            ax1.scatter(xpos + rng.uniform(-0.04, 0.04, len(frac)), frac, s=4, marker=MARKERS[m], color=COLORS[m],
                        alpha=0.8, linewidths=0, zorder=3)
            Z = np.vstack([feature_mask(h, F) for h in runs.best_selected_mask_hex])
            stab, lo, hi = nogueira_stability(Z)
            ax2.errorbar(xpos, stab, yerr=[[stab - lo], [hi - stab]], fmt=MARKERS[m], color=COLORS[m], markersize=4,
                         elinewidth=1.0, capsize=0, markeredgecolor="white", markeredgewidth=0.4)
            rows.append({"dataset": ds, "method": m, "F": F, "median_n_selected": float(np.median(runs.n_selected)),
                         "stability": stab, "ci_low": lo, "ci_high": hi})
    ax1.axhline(0.5, color="#BDBDBD", linewidth=0.6, zorder=1)
    ax1.text(-0.52, 0.06, "grey line: expected fraction for uniform random genes (0.5)", fontsize=6.5,
             color=INK_MUTED, ha="left", va="bottom")
    ax1.set_ylabel("Selected features\n(fraction of F)")
    ax1.yaxis.grid(True, **GRID)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Nogueira stability Φ\n(95% CI)")
    ax2.yaxis.grid(True, **GRID)
    F_by = {r["dataset"]: r["F"] for r in rows}
    ax2.set_xticks(range(len(DATASETS)), [f"{ds}\n(F = {F_by[ds]})" for ds in DATASETS])
    ax2.set_xlim(-0.55, len(DATASETS) - 0.45)
    for idx, ax in enumerate((ax1, ax2)):
        panel_label(ax, idx, x=-0.09)
    fig.legend(handles=_method_handles(include_default=False), loc="upper center", ncol=5, bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.92, bottom=0.11, hspace=0.18)
    return _finish(fig, study, "fig6_feature_counts", out), rows


# ----------------------------------------------------------------------------------------------------------
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


def figS_cd_diagrams(study: Study, out: Path) -> tuple[list[Path], list[dict]]:
    """Critical-difference diagrams across datasets (Demsar 2006), one per experiment. Low-powered: N = 5."""
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL, 3.6))
    rows = []
    for idx, exp in enumerate(EXPERIMENTS):
        good = np.array([[np.median(_goodness(ds, _values(_cell(study, ds, exp), m))) for m in METHODS]
                         for ds in DATASETS])
        ranks = average_ranks(good)
        cd = nemenyi_cd(len(METHODS), len(DATASETS))
        p = friedmanchisquare(*good.T).pvalue
        _draw_cd(axes[idx], list(METHODS), ranks, cd,
                 f"Experiment {exp}: mean rank over N = {len(DATASETS)} datasets (low-powered)\n"
                 f"Nemenyi CD at α = 0.05; Friedman p = {p:.2f}")
        panel_label(axes[idx], idx, x=0.0, y=0.93)
        rows += [{"experiment": exp, "method": m, "mean_rank": float(r), "CD": cd, "friedman_p": float(p)}
                 for m, r in zip(METHODS, ranks)]
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02, hspace=0.15)
    return _finish(fig, study, "figS_cd_diagrams", out), rows


# ----------------------------------------------------------------------------------------------------------
def figS_a12_heatmap(study: Study, out: Path) -> list[Path]:
    """Vargha-Delaney A12 of row vs column optimizer (test primary metric, 30 seeds)."""
    cmap = LinearSegmentedColormap.from_list("div", ["#B2182B", "#F2F2F2", "#2166AC"])  # col better | neutral | row better
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(DOUBLE_COL, 3.7))
    n = len(OPTIMIZERS)
    im = None
    for i, exp in enumerate(EXPERIMENTS):
        for j, ds in enumerate(DATASETS):
            ax, sub = axes[i, j], _cell(study, ds, exp)
            vals = {m: _goodness(ds, _values(sub, m)) for m in OPTIMIZERS}
            mat = np.full((n, n), np.nan)
            for r, mr in enumerate(OPTIMIZERS):
                for c, mc in enumerate(OPTIMIZERS):
                    if r != c:
                        mat[r, c] = a12(vals[mr], vals[mc])
            im = ax.imshow(mat, cmap=cmap, vmin=0.2, vmax=0.8)  # NaN diagonal renders as the white background
            for r in range(n):
                for c in range(n):
                    if r != c:
                        v = mat[r, c]
                        ax.text(c, r, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.5,
                                color="white" if abs(v - 0.5) > 0.22 else "black")
            ax.set_xticks(range(n), OPTIMIZERS if i == 1 else [""] * n, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticks(range(n), OPTIMIZERS if j == 0 else [""] * n)
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
    fig.subplots_adjust(left=0.10, right=0.91, top=0.92, bottom=0.14, wspace=0.12, hspace=0.18)
    return _finish(fig, study, "figS_a12_heatmap", out)
