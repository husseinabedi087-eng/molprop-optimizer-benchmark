"""Fast release check (seconds, no model fitting). Usage: python validate_release.py

Verifies, from the files in this repository:
  1. scaffold-split identities (SHA-256 recorded at preparation) and split sizes (Table 1);
  2. search dimensions of Experiment B (8 + F = 196, 179, 215, 218, 201) from every run file;
  3. budgets: every primary, mirror and revision run used exactly its budget, with 30 seeds per primary cell;
  4. code hashes: each run batch carries one code hash; the revision code hash matches the current code;
  5. pre-registration hashes (mirror check, revision plan, amendment);
  6. the exact-reproduction flags of the revision re-evaluation (J1).
"""
import hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES, REV = ROOT / "results" / "xgb_mh", ROOT / "results" / "revision"
DIMS = {"ESOL": 196, "FreeSolv": 179, "Lipophilicity": 215, "BBBP": 218, "BACE": 201}
SIZES = {"ESOL": (902, 113, 113), "FreeSolv": (513, 64, 65), "Lipophilicity": (3360, 420, 420), "BBBP": (1631, 204, 204),
         "BACE": (1210, 151, 152)}
BATCH = {"full": "4669997af4a6", "full_ref": "f056a88816b3", "mirror": "fc7d77de4416", "revision": "e0abd2b465a1"}
ok = True

def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok &= bool(cond)

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary = json.loads((RES / "features" / "summary.json").read_text())
for ds, (ntr, nva, nte) in SIZES.items():
    split = json.loads((RES / "splits" / f"{ds}.json").read_text())
    h = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    check(h == summary[ds]["split"]["sha256"], f"{ds}: split hash {h[:12]} matches the recorded hash")
    fin = summary[ds]["final_split_sizes"]
    check((fin["train"], fin["validation"], fin["test"]) == (ntr, nva, nte), f"{ds}: split sizes {ntr}/{nva}/{nte} (Table 1)")

runs = {}
for batch in ("full", "full_ref", "mirror"):
    runs[batch] = [json.loads(p.read_text()) for p in sorted((RES / batch / "runs").glob("*.json"))]
dims = Counter()
for batch, rs in runs.items():
    for r in rs:
        if r["experiment"] == "B":
            d = 8 + r["n_features_total"]; dims[(r["dataset"], d)] += 1
for ds, d in DIMS.items():
    seen = {dd for (dds, dd) in dims if dds == ds}
    check(seen == {d}, f"{ds}: Experiment B dimension 8 + F = {d} in all run files")
check(min(d for _, d in dims) == 179 and max(d for _, d in dims) == 218, "Experiment B dimensions range 179-218")

for batch, rs in runs.items():
    check(all(r["n_calls"] == r["budget"] == 100 for r in rs), f"{batch}: all {len(rs)} runs used exactly 100 calls")
    meta = json.loads((RES / batch / "run_metadata.json").read_text())
    check(meta["code_sha256"].startswith(BATCH[batch]), f"{batch}: code hash {BATCH[batch]}")
prim = Counter((r["dataset"], r["experiment"], r["optimizer"]) for b in ("full", "full_ref") for r in runs[b])
check(all(v == 30 for v in prim.values()) and len(prim) == 70, "primary runs: 30 seeds in each of 70 dataset x experiment x optimizer cells")

if REV.exists():
    sys.path.insert(0, str(ROOT))
    from revision.jobs import code_hash
    ch = code_hash()
    check(ch.startswith(BATCH["revision"]), f"revision: current code hash {ch[:12]} equals the hash of the revision runs")
    rev = [json.loads(p.read_text()) for p in REV.glob("*/*.json")]
    check(all(r.get("code_sha256", "").startswith(BATCH["revision"]) for r in rev), f"revision: all {len(rev)} result files carry code hash {BATCH['revision']}")
    srch = [r for r in rev if r.get("job") in ("J3", "J4", "J5")]
    check(all(r["n_calls"] == r["budget"] for r in srch) and len(srch) == 1040, f"revision searches: {len(srch)} runs, each used exactly its budget")
    j6 = [r for r in rev if r.get("job") == "J6" and r.get("optimizer") != "default"]
    check(all(r["n_calls"] == 100 for r in j6) and len(j6) == 300, f"revision partitions: {len(j6)} runs of 100 calls")
    j1 = [r for r in rev if r.get("job") == "J1" and "refit_reproduces_primary_exactly" in r]
    check(len(j1) == 1500 and all(r["refit_reproduces_primary_exactly"] and r["val_reproduces_logged_exactly"] for r in j1),
          f"revision J1: {sum(r['refit_reproduces_primary_exactly'] for r in j1)}/1500 refits reproduce the published test scores exactly")
    check(len([r for r in rev if r.get("job") == "J2"]) == 1005, "revision validation control: 1005 pool evaluations")

for md, rec in ((RES / "mirror" / "prediction.md", RES / "mirror" / "prediction.sha256"),
                (ROOT / "revision" / "PREREGISTRATION_REVISION.md", ROOT / "revision" / "PREREGISTRATION_REVISION.sha256"),
                (ROOT / "revision" / "AMENDMENT_1.md", ROOT / "revision" / "AMENDMENT_1.sha256")):
    if md.exists():
        check(sha(md) == rec.read_text().split()[0], f"{md.relative_to(ROOT)}: SHA-256 matches its record")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
