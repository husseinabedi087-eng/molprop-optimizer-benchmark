"""Progress and sanity report. Usage: python -m revision.status"""
import json
from pathlib import Path
from .core import OUT
from .jobs import tasks_for, out_name

for job in ("J1", "J3", "J4", "J5", "J2", "J6"):
    t = tasks_for(job)
    done = [o for o in (out_name(*x) for x in t) if o.with_name(o.name + ".json").exists()]
    print(f"{job}: {len(done)}/{len(t)} done")
    if job == "J1" and done:
        rows = [json.loads(o.with_name(o.name + ".json").read_text()) for o in done]
        r = [x for x in rows if "refit_reproduces_primary_exactly" in x]
        if r:
            print(f"    refit test score identical to the published run: {sum(x['refit_reproduces_primary_exactly'] for x in r)}/{len(r)}")
            print(f"    validation score identical to the published log:  {sum(x['val_reproduces_logged_exactly'] for x in r)}/{len(r)}")
