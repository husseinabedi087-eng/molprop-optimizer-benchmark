"""Build the FINAL figure set + contact sheet. Read-only over results/ and analysis/final/.

Usage (from the project root, low priority):  python analysis/figures/make_figures.py
Outputs: analysis/figures/out/<figure>.{pdf,png,tiff}, contact_sheet.png, figure_data.json
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import contact_sheet
import figures as F
import style
from check_palette import report as palette_report
from data import (DATASETS, EXPERIMENTS, load_a0_runs, load_curves, load_final_stats, load_mirror_pairs,
                  load_mirror_tests, load_study)

OUT = Path(__file__).resolve().parent / "out"
# final_stats.json was written with pandas to_json (10 decimal places), so exact agreement is to ~5e-11.
JSON_TOL = 1e-9


def verify_against_final_stats(study, fs, pairs, mtests, a0) -> dict:
    """The raw data drawn as distributions must reproduce the statistics in final_stats.json exactly."""
    checks = {}
    tvd = {(r["dataset"], r["experiment"], r["method"]): r for r in fs["tuning_vs_default"] if r["role"] == "primary"}
    ok = all(np.isclose(np.median(F._values(F._cell(study, d, e), m)), r["median"], rtol=0, atol=JSON_TOL)
             and np.isclose(F._values(F._cell(study, d, e), "Default")[0], r["default"], rtol=0, atol=JSON_TOL)
             for (d, e, m), r in tvd.items())
    checks["fig2/fig3: raw test values reproduce tuning_vs_default medians and default values"] = bool(ok)
    vr = fs["validation_reliability"]
    checks["fig4: raw val/test values reproduce validation_reliability medians"] = bool(all(
        np.isclose(np.median(F._values(F._cell(study, r["dataset"], r["experiment"]), r["method"], "val_primary")),
                   r["median_val"], rtol=0, atol=JSON_TOL) for r in vr))
    fsel = fs["feature_selection"]
    checks["fig5a: raw fractions reproduce feature_selection medians"] = bool(all(
        np.isclose(np.median((lambda s: s.n_selected / s.n_features_total)(
            study.runs[(study.runs.dataset == r["dataset"]) & (study.runs.experiment == "B") & (study.runs.method == r["method"])])),
            r["fraction_median"], rtol=0, atol=JSON_TOL) for r in fsel))
    mt = mtests[mtests.outcome == "fraction"]
    checks["fig5c: paired raw fractions reproduce the pre-registered mirror medians"] = bool(all(
        np.isclose(np.median(pairs[(pairs.dataset == r.dataset) & (pairs.method == r.method)].frac_normal), r.median_normal,
                   atol=JSON_TOL) and
        np.isclose(np.median(pairs[(pairs.dataset == r.dataset) & (pairs.method == r.method)].frac_mirrored),
                   r.median_mirrored, atol=JSON_TOL) for r in mt.itertuples()))
    checks["figS5_sensitivity: raw a0/ref values reproduce sensitivity_paired medians"] = bool(all(
        np.isclose(np.median(a0[(a0.dataset == r["dataset"]) & (a0.experiment == r["experiment"]) & (a0.method == r["method"])]
                             .test_primary), r["median_a0"], atol=JSON_TOL) and
        np.isclose(np.median(F._values(F._cell(study, r["dataset"], r["experiment"]), r["method"])), r["median_ref"], atol=JSON_TOL)
        for r in fs["sensitivity_paired"]))
    checks["no DRAFT variants: GWO/WOA are gwo_ref/woa_ref"] = not study.draft
    return checks


def main() -> None:
    t0 = time.perf_counter()
    style.apply()
    study, fs = load_study(), load_final_stats()
    pairs, mtests, a0 = load_mirror_pairs(study), load_mirror_tests(), load_a0_runs()
    checks = verify_against_final_stats(study, fs, pairs, mtests, a0)
    if not all(checks.values()):
        raise SystemExit(f"Raw data disagree with final_stats.json: {checks}")
    curves = load_curves(study)
    written = {
        "fig2_test_distributions": F.fig2_test_distributions(study, fs, OUT),
        "fig3_forest_vs_default": F.fig3_forest_vs_default(study, fs, OUT),
        "fig4_validation_reliability": F.fig4_validation_reliability(study, fs, OUT),
        "fig5_feature_selection": F.fig5_feature_selection(study, fs, pairs, mtests, OUT),
        "figS1_sphere": F.figS1_sphere(OUT),
        "figS2_a12_heatmap": F.figS2_a12_heatmap(fs, OUT),
        "figS3_cd_optimizers": F.figS3_cd_optimizers(fs, OUT),
        "figS4_headroom_EXPLORATORY": F.figS4_headroom_exploratory(fs, OUT),
        "figS5_sensitivity": F.figS5_sensitivity(study, a0, fs, OUT),
        "figS6_convergence": F.figS6_convergence(study, curves, fs, OUT),
    }  # manuscript order (also the contact-sheet order)
    # Print-size text audit: figures are saved at their print width, so these are the printed point sizes.
    widths_ok = all(a["width_in"] in (style.SINGLE_COL, style.DOUBLE_COL) for a in style.TEXT_AUDIT.values())
    text_ok = all(max(a["font_sizes_pt"]) <= 9 and min(a["font_sizes_pt"]) >= 6.5 for a in style.TEXT_AUDIT.values())
    checks["print size: every figure is 3.5 in or 7.2 in wide"] = widths_ok
    checks["text: body 8 pt, ticks/legend 7 pt, annotations >= 6.5 pt, panel letters 9 pt (none larger)"] = text_ok
    if not (widths_ok and text_ok):
        raise SystemExit(f"Text/size audit failed: {style.TEXT_AUDIT}")
    sheet = contact_sheet.build([paths[1] for paths in written.values()], OUT / "contact_sheet.png",
                                f"xgb_mh FINAL figures - GWO/WOA = gwo_ref/woa_ref - {datetime.now():%Y-%m-%d %H:%M}")
    sources = {
        "statistics": "analysis/final/final_stats.json (analysis code sha256 "
                      f"{fs['provenance']['analysis_code_sha256']})",
        "mirror tests (fig5c)": "analysis/final/5_mirror_prespecified_tests.csv (= RESULTS.md section 5)",
        "sphere (figS1_sphere)": "results/xgb_mh/sanity/sphere_tests.csv, sphere_corner_bias.csv",
        "raw distributions": "results/xgb_mh/full (Random, TPE, PSO, Default), full_ref (GWO, WOA), mirror, "
                             "full gwo/woa (a0, figS5_sensitivity only); call logs for figS6_convergence",
    }
    (OUT / "figure_data.json").write_text(json.dumps({
        "generated": datetime.now().isoformat(timespec="seconds"), "draft": study.draft, "sources": sources,
        "consistency_checks": checks, "text_audit": style.TEXT_AUDIT,
        "files": {k: [p.name for p in v] for k, v in written.items()},
        "palette_cvd_check": [{"condition": c, "closest_pair": f"{a} vs {b}", "min_delta_e": round(d, 1)}
                              for c, a, b, d in palette_report()]}, indent=2), encoding="utf-8")
    for k, v in checks.items():
        print("CHECK", "PASS" if v else "FAIL", "-", k)
    for name, a in style.TEXT_AUDIT.items():
        print(f"TEXT {name:28s} {a['width_in']} x {a['height_in']} in | font sizes (pt): {a['font_sizes_pt']}")
    for name, paths in written.items():
        print(name, *[p.name for p in paths])
    print("contact sheet:", sheet)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
