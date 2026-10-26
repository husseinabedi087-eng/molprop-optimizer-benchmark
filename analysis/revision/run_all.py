"""Regenerate every revision table and figure (analysis/revision/out, analysis/revision/fig) from the results.
Usage: python analysis/revision/run_all.py   (about 10 minutes; needs the unzipped revision call logs for existing2.py)"""
import runpy
from pathlib import Path
HERE = Path(__file__).resolve().parent
for s in ("exact_recompute", "existing", "existing2", "j1", "j1_parents", "j3_j4", "j5", "j2_j6", "figures_rev"):
    print(f"== {s}.py", flush=True)
    runpy.run_path(str(HERE / f"{s}.py"), run_name="__main__")
