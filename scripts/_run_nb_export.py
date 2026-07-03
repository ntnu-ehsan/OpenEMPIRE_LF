"""Execute scenarios_comparison.ipynb code cells in-process to (re)export PNGs.

Run with a python that has kaleido installed, from the scripts/ directory so the
notebook's relative paths (../Results, ../plots) resolve.
"""
import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent / "scenarios_comparison.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    print(f"[cell {i}] running...", flush=True)
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    except Exception as e:
        print(f"  ERROR in cell {i}: {type(e).__name__}: {e}", flush=True)
        raise
print("DONE all cells", flush=True)
