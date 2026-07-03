"""Execute results_reviewer.ipynb cells for a given DATASET_NAME, saving plots.

Runs each code cell in a shared namespace with a non-interactive backend.
IPython magics and shell escapes are stripped; ``display`` is a no-op.
Continues on per-cell errors and reports a summary so we get as many plots
as possible.
"""
import sys
import re
import json
import traceback

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET = sys.argv[1]
NB_PATH = "results_reviewer.ipynb"

nb = json.load(open(NB_PATH, encoding="utf-8"))

g = {"__name__": "__main__"}
g["display"] = lambda *a, **k: None  # dataframe styler/display -> no-op

failures = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not src.strip():
        continue
    # strip ipython line magics and shell escapes
    src = "\n".join(
        ln for ln in src.split("\n")
        if not ln.lstrip().startswith("%") and not ln.lstrip().startswith("!")
    )
    # force the dataset selection in the config cell
    src = re.sub(r'^DATASET_NAME\s*=.*$',
                 f'DATASET_NAME = {DATASET!r}',
                 src, count=1, flags=re.MULTILINE)
    # pre-existing bug: only 4 colors zipped against 7 periods -> KeyError.
    # Extend the palette so every investment period gets a colour.
    src = src.replace(
        "['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']",
        "['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', "
        "'#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']")
    try:
        exec(compile(src, f"cell_{i}", "exec"), g)
        plt.close("all")
    except Exception:
        failures.append(i)
        print(f"[cell {i}] FAILED:")
        traceback.print_exc()
        plt.close("all")

print("\n==== DONE for", DATASET, "====")
print("Failed cells:", failures if failures else "none")
