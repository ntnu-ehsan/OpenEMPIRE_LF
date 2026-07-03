"""Run scenarios_comparison.ipynb cells but capture each figure as JSON instead of
calling kaleido (which hangs in batch). Figures are dumped to scripts/_figjson/.

Run from scripts/ with any python (no kaleido needed)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "scenarios_comparison.ipynb"
FIGDIR = HERE / "_figjson"
FIGDIR.mkdir(exist_ok=True)

nb = json.loads(NB.read_text(encoding="utf-8"))
ns = {"__name__": "__main__"}

dumped = []

def _save_json(fig, name):
    try:
        (FIGDIR / f"{name}.json").write_text(fig.to_json(), encoding="utf-8")
        dumped.append(name)
        print(f"  dumped {name}", flush=True)
    except Exception as e:
        print(f"  FAILED dump {name}: {e}", flush=True)

def _show(fig, name):
    _save_json(fig, name)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    exec(compile(src, f"<cell {i}>", "exec"), ns)
    # After the helper cell defines save/show_interactive, override them so no
    # figure cell triggers kaleido; they just dump JSON.
    if "def show_interactive" in src:
        ns["save"] = _save_json
        ns["show_interactive"] = _show

print(f"DONE. dumped {len(dumped)} figures: {dumped}", flush=True)
