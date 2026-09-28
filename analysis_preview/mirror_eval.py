"""Mirrored-encoding check: audit + evaluation EXACTLY as pre-registered in results/xgb_mh/mirror/prediction.md.

Read-only over results/. Imports xgb_mh.space / optimizers._common / experiment only to recompute the
mirrored initial design (run with PYTHONDONTWRITEBYTECODE=1 so nothing is written into xgb_mh/).
Writes analysis_preview/output/mirror_<stamp>/.
Usage: python analysis_preview/mirror_eval.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from xgb_mh.optimizers._common import initial_design  # noqa: E402  (read-only use)
from xgb_mh.space import decode, mask_to_hex, mirror  # noqa: E402

RES = ROOT / "results" / "xgb_mh"
MIRROR, FULL, FULL_REF = RES / "mirror", RES / "full", RES / "full_ref"
OUT = Path(__file__).resolve().parent / "output"
DATASETS = ("BBBP", "FreeSolv")
OPTIMIZERS = ("random", "tpe", "gwo_ref", "woa_ref", "pso")
NORMAL_SOURCE = {"random": FULL, "tpe": FULL, "pso": FULL, "gwo_ref": FULL_REF, "woa_ref": FULL_REF}
PRIMARY = {"BBBP": "test_roc_auc", "FreeSolv": "test_rmse"}
SEEDS = range(30)
N_INITIAL, BUDGET, N_HP = 10, 100, 8


def holm(p: list[float]) -> list[float]:
    order, adj, running, m = np.argsort(p), [0.0] * len(p), 0.0, len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def fp(p: float) -> str:
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def table(header, rows) -> str:
    return "\n".join("| " + " | ".join(map(str, r)) + " |" for r in [header, ["---"] * len(header)] + rows)


def main() -> None:
    out = OUT / f"mirror_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    rep = ["# Mirrored-encoding robustness check", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Read-only over results/xgb_mh/{{mirror,full,full_ref}}.", ""]

    # ---- 1. pre-registration integrity ----
    rec = (MIRROR / "prediction.sha256").read_text().split()[0]
    now = hashlib.sha256((MIRROR / "prediction.md").read_bytes()).hexdigest()
    meta = json.loads((MIRROR / "run_metadata.json").read_text())
    rep += ["## 1. Pre-registration", "", f"- prediction.md sha256 matches prediction.sha256: **{rec == now}** (`{now[:16]}...`)",
            f"- mirror run created (UTC) {meta['created_utc']}; recorded config: {meta['config']}", ""]

    # ---- 2. audit ----
    runs = {}
    problems = []
    design_ok = design_n = 0
    for p in sorted((MIRROR / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        runs[(r["dataset"], r["experiment"], r["optimizer"], r["seed"])] = r
        if r.get("encoding") != "mirrored" or "mirrored encoding" not in json.loads(r["optimizer_meta"]).get("encoding", ""):
            problems.append(f"{p.stem}: encoding not logged as mirrored")
        with (MIRROR / "calls" / f"{p.stem}.jsonl").open() as handle:
            calls = [json.loads(line) for line in handle]
        if [c["call_idx"] for c in calls] != list(range(BUDGET)) or r["n_calls"] != BUDGET:
            problems.append(f"{p.stem}: not exactly 100 ordered calls")
        F = r["n_features_total"]
        design = initial_design(np.random.default_rng(r["seed"]), N_HP + F)
        for c, x in zip(calls[:N_INITIAL], design):
            cfg = decode(mirror(x), F)
            design_n += 1
            design_ok += c["config"] == cfg.params and c["selected_mask_hex"] == mask_to_hex(cfg.selected, F)
    expected = {(d, "B", o, s) for d in DATASETS for o in OPTIMIZERS for s in SEEDS}
    # Calls 1-10 also carry identical fitness across the 5 optimizers (same decoded config, same seed).
    fit_shared = 0
    for d in DATASETS:
        for s in SEEDS:
            f10 = {o: [json.loads(line)["fitness"] for line in list((MIRROR / "calls" / f"{d}_B_{o}_s{s}.jsonl").open())[:N_INITIAL]]
                   for o in OPTIMIZERS}
            fit_shared += len({tuple(v) for v in f10.values()}) == 1
    audit_ok = (set(runs) == expected and not problems and design_ok == design_n and fit_shared == 60
                and not (MIRROR / "errors.log").exists() and meta["config"].get("encoding") == "mirrored")
    rep += ["## 2. Audit of the mirror run", "",
            f"- **{'PASS' if audit_ok else 'FLAG'}**: {len(runs)} runs (expected 300, missing {len(expected - set(runs))}, "
            f"unexpected {len(set(runs) - expected)}); every run 100 ordered calls and `encoding: mirrored`: {not problems}; "
            f"calls 1-10 decode to decode(1 - shared design) in {design_ok}/{design_n} calls (config + feature mask, exact); "
            f"calls 1-10 fitness identical across the 5 optimizers in {fit_shared}/60 (dataset, seed) groups; "
            f"errors.log {'PRESENT' if (MIRROR / 'errors.log').exists() else 'absent'}"
            + (f"; problems: {problems[:3]}" if problems else ""), ""]

    # ---- 3. pre-specified evaluation ----
    mir = pd.read_csv(MIRROR / "test_summary.csv")
    mir = mir[mir.optimizer != "default"]
    summaries = {src: pd.read_csv(src / "test_summary.csv") for src in set(NORMAL_SOURCE.values())}
    norm = pd.concat([summaries[src][(summaries[src].optimizer == o) & (summaries[src].experiment == "B")]
                      for o, src in NORMAL_SOURCE.items()])
    rows = []
    for d in DATASETS:
        pm = PRIMARY[d]
        for o in OPTIMIZERS:
            a = norm[(norm.dataset == d) & (norm.optimizer == o)].set_index("seed").sort_index()
            b = mir[(mir.dataset == d) & (mir.optimizer == o)].set_index("seed").sort_index()
            assert list(a.index) == list(b.index) == list(SEEDS), (d, o)
            fa, fb = a.n_selected / a.n_features_total, b.n_selected / b.n_features_total
            for outcome, x, y in (("fraction", fa, fb), (pm, a[pm], b[pm])):
                diff = (y - x).to_numpy()
                p = 1.0 if np.all(diff == 0) else float(wilcoxon(y, x, zero_method="pratt").pvalue)
                rows.append({"dataset": d, "optimizer": o, "outcome": outcome, "median_normal": float(np.median(x)),
                             "median_mirrored": float(np.median(y)), "median_paired_diff": float(np.median(diff)),
                             "n_mirrored_higher": int((diff > 0).sum()), "p_raw": p})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for (d, outcome), g in res.groupby(["dataset", "outcome"]):
        res.loc[g.index, "p_holm"] = holm(list(g.p_raw))
    res.to_csv(out / "prespecified_tests.csv", index=False)

    # Decision rule, as written in prediction.md.
    def gwo_ok(d):
        r = res[(res.dataset == d) & (res.optimizer == "gwo_ref") & (res.outcome == "fraction")].iloc[0]
        return r.p_holm < 0.05 and r.median_paired_diff > 0, r.median_mirrored > 0.5, r
    checks = {d: gwo_ok(d) for d in DATASETS}
    n_met = sum(int(bool(c[0])) + int(bool(c[1])) for c in checks.values())  # numpy bool + bool is OR, not 2
    gwo_verdict = "SUPPORTED" if n_met == 4 else ("NOT SUPPORTED" if n_met == 0 else "PARTLY SUPPORTED")
    rand = res[(res.optimizer == "random") & (res.outcome == "fraction")]
    rand_verdict = "SUPPORTED (no change)" if (rand.p_holm >= 0.05).all() else "NOT SUPPORTED (Random's fraction changed)"

    rep += ["## 3. Pre-specified tests", "",
            "Paired Wilcoxon signed-rank (two-sided, zero_method = pratt), pairs = 30 tuning seeds, normal vs mirrored; "
            "Holm over the 5 optimizers within each dataset x outcome. diff = mirrored - normal.", ""]
    for d in DATASETS:
        for outcome in ("fraction", PRIMARY[d]):
            g = res[(res.dataset == d) & (res.outcome == outcome)]
            label = "selected-feature fraction" if outcome == "fraction" else outcome.replace("test_", "test ")
            rep += [f"**{d} - {label}**", "",
                    table(["optimizer", "median normal", "median mirrored", "median paired diff", "seeds mirrored > normal",
                           "p raw", "p Holm"],
                          [[r.optimizer, f"{r.median_normal:.3f}", f"{r.median_mirrored:.3f}", f"{r.median_paired_diff:+.3f}",
                            f"{r.n_mirrored_higher}/30", fp(r.p_raw), fp(r.p_holm) + (" *" if r.p_holm < 0.05 else "")]
                           for r in g.itertuples()]), ""]

    rep += ["## Verdict (decision rule from prediction.md)", "",
            "gwo_ref supports the prediction if, on EACH dataset, its mirrored fraction is higher (Holm p < 0.05) AND its "
            "median mirrored fraction is > 0.5. random (control) supports it if its fraction shows no Holm-significant "
            "difference on either dataset. (The rule does not define 'partly'; used here: some but not all of the 4 "
            "gwo_ref conditions met.)", ""]
    for d, (higher, above, r) in checks.items():
        rep.append(f"- gwo_ref {d}: mirrored fraction higher with Holm p < 0.05: **{higher}** (median diff "
                   f"{r.median_paired_diff:+.3f}, p Holm {fp(r.p_holm)}); median mirrored fraction > 0.5: **{above}** "
                   f"({r.median_mirrored:.3f})")
    rep += ["", f"**gwo_ref: {gwo_verdict}** ({n_met}/4 conditions met).",
            f"**random (control): {rand_verdict}** (p Holm: " + ", ".join(f"{r.dataset} {fp(r.p_holm)}" for r in rand.itertuples()) + ").", ""]

    # ---- 4. exploratory ----
    expl = []
    for d in DATASETS:
        for o in OPTIMIZERS:
            a = norm[(norm.dataset == d) & (norm.optimizer == o)].set_index("seed").sort_index()
            b = mir[(mir.dataset == d) & (mir.optimizer == o)].set_index("seed").sort_index()
            expl.append({"dataset": d, "optimizer": o,
                         "val_raw_normal": a.best_val_raw_metric.median(), "val_raw_mirrored": b.best_val_raw_metric.median(),
                         "unique_normal": a.n_unique.median(), "unique_mirrored": b.n_unique.median(),
                         "frac_iqr_normal": f"{(a.n_selected / a.n_features_total).quantile(.25):.2f}-"
                                            f"{(a.n_selected / a.n_features_total).quantile(.75):.2f}",
                         "frac_iqr_mirrored": f"{(b.n_selected / b.n_features_total).quantile(.25):.2f}-"
                                              f"{(b.n_selected / b.n_features_total).quantile(.75):.2f}"})
    ex = pd.DataFrame(expl)
    ex.to_csv(out / "exploratory.csv", index=False)
    rep += ["## 4. EXPLORATORY (not pre-specified; no inference)", "",
            table(["dataset", "optimizer", "fraction IQR normal", "fraction IQR mirrored", "median val raw normal",
                   "median val raw mirrored", "median unique normal", "median unique mirrored"],
                  [[r.dataset, r.optimizer, r.frac_iqr_normal, r.frac_iqr_mirrored, f"{r.val_raw_normal:.4f}",
                    f"{r.val_raw_mirrored:.4f}", f"{r.unique_normal:.0f}", f"{r.unique_mirrored:.0f}"] for r in ex.itertuples()]),
            "", "val raw = best validation raw metric (RMSE for FreeSolv, 1 - ROC-AUC for BBBP; lower is better).", ""]
    text = "\n".join(rep)
    (out / "MIRROR_EVAL.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[written to {out}]")


if __name__ == "__main__":
    main()
