"""Generate a PowerPoint summarizing original EMPIRE and the fork's changes."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ---- palette ----
DARK = RGBColor(0x1F, 0x2A, 0x44)      # deep navy
ACCENT = RGBColor(0x1B, 0x8A, 0x8F)    # teal
ACCENT2 = RGBColor(0xE8, 0x7A, 0x2B)   # orange
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
GREY = RGBColor(0x55, 0x5B, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, color, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tb.text_frame.word_wrap = True
    return tb


def set_run(run, text, size, color, bold=False, italic=False, font="Calibri"):
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font


def title_bar(slide, title, kicker=None):
    rect(slide, 0, 0, SW, Inches(1.15), DARK)
    rect(slide, 0, Inches(1.15), SW, Inches(0.06), ACCENT)
    tb = textbox(slide, Inches(0.55), Inches(0.18), Inches(12.2), Inches(0.95))
    tf = tb.text_frame
    if kicker:
        p = tf.paragraphs[0]
        set_run(p.add_run(), kicker.upper(), 12, ACCENT2, bold=True)
        p2 = tf.add_paragraph()
        set_run(p2.add_run(), title, 28, WHITE, bold=True)
    else:
        p = tf.paragraphs[0]
        set_run(p.add_run(), title, 30, WHITE, bold=True)


def code_box(slide, x, y, w, h, lines, size=11.5, title=None, bg=RGBColor(0x2B, 0x2F, 0x3A)):
    """Monospace code/listing box with a dark background."""
    box = rect(slide, x, y, w, h, bg)
    pad = Inches(0.18)
    ty = y + pad
    if title:
        ttb = textbox(slide, x + pad, y + Inches(0.04), w - 2 * pad, Inches(0.3))
        set_run(ttb.text_frame.paragraphs[0].add_run(), title, size - 1.5,
                RGBColor(0x8A, 0xD8, 0xC8), bold=True, font="Consolas")
        ty = y + Inches(0.42)
    tb = textbox(slide, x + pad, ty, w - 2 * pad, h - (ty - y) - Inches(0.1))
    tf = tb.text_frame
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(1)
        if isinstance(ln, tuple):
            text, color = ln
        else:
            text, color = ln, RGBColor(0xE6, 0xE9, 0xEF)
        set_run(p.add_run(), text, size, color, font="Consolas")
    return box


def _emit_math(p, text, size, color, bold):
    """Render math markup: _{...} = subscript, ^{...} = superscript."""
    i, n = 0, len(text)
    while i < n:
        if text[i] in "_^" and i + 1 < n and text[i + 1] == "{":
            j = text.index("}", i + 2)
            run = p.add_run()
            set_run(run, text[i + 2:j], size * 0.7, color, bold=bold, font="Cambria Math")
            rPr = run._r.get_or_add_rPr()
            rPr.set("baseline", "-25000" if text[i] == "_" else "30000")
            i = j + 1
        else:
            k, buf = i, ""
            while k < n and not (text[k] in "_^" and k + 1 < n and text[k + 1] == "{"):
                buf += text[k]
                k += 1
            run = p.add_run()
            set_run(run, buf, size, color, bold=bold, font="Cambria Math")
            i = k


def math_box(slide, x, y, w, h, lines, size=18, bg=LIGHT):
    """Light box for a mathematical expression. lines: str or (str, size, color, bold)."""
    rect(slide, x, y, w, h, bg)
    rect(slide, x, y, Inches(0.10), h, ACCENT2)
    tb = textbox(slide, x + Inches(0.3), y + Inches(0.12), w - Inches(0.5), h - Inches(0.2))
    tf = tb.text_frame
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        if isinstance(ln, tuple):
            text, sz, col, bold = ln
        else:
            text, sz, col, bold = ln, size, DARK, False
        _emit_math(p, text, sz, col, bold)
    return tb


def bullets(slide, items, x, y, w, h, size=16, gap=6):
    tb = textbox(slide, x, y, w, h)
    tf = tb.text_frame
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p.level = level
        bullet = "•  " if level == 0 else "–  "
        r = p.add_run()
        set_run(r, bullet + text, size - level * 1, DARK if level == 0 else GREY,
                bold=(level == 0))
    return tb


# ======================================================================
# Slide 1 — Title
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(4.55), SW, Inches(0.08), ACCENT)
rect(s, 0, Inches(4.63), SW, Inches(0.04), ACCENT2)
tb = textbox(s, Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.5))
tf = tb.text_frame
p = tf.paragraphs[0]
set_run(p.add_run(), "OpenEMPIRE — North Sea Extension", 40, WHITE, bold=True)
p2 = tf.add_paragraph()
set_run(p2.add_run(), "From the original capacity-expansion model to a "
        "performance-tuned North Sea build with offshore nodes", 20,
        RGBColor(0xC9, 0xD2, 0xE0))
tb2 = textbox(s, Inches(0.8), Inches(4.8), Inches(11.7), Inches(1.2))
tf2 = tb2.text_frame
p = tf2.paragraphs[0]
set_run(p.add_run(), "Ehsan Nokandi", 18, ACCENT2, bold=True)
p2 = tf2.add_paragraph()
set_run(p2.add_run(), "Summary of upstream EMPIRE and the North Sea modifications in "
        "this fork", 14, RGBColor(0xA9, 0xB4, 0xC4))

# ======================================================================
# Slide 2 — What is EMPIRE (overview)
# ======================================================================
s = add_slide()
title_bar(s, "What is EMPIRE?", "Original model")
bullets(s, [
    "A stochastic capacity-expansion model for the European power system, "
    "formulated in Pyomo (open-source, MIT-licensed).",
    "Co-optimizes long-run investment and short-run operation under uncertainty "
    "to minimize total discounted system cost.",
    "Two-stage stochastic program: investment decisions per strategic period, "
    "operational decisions across sampled scenarios.",
    "Multi-horizon: strategic investment periods (e.g. every 5 years to 2060) "
    "each containing representative operational seasons + peak hours.",
    "Solved with a third-party LP solver (Gurobi / Xpress / CPLEX); a full run "
    "needs an HPC (~140 GB RAM).",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.5), size=17, gap=10)

# ======================================================================
# Slide 3 — How EMPIRE works (structure)
# ======================================================================
s = add_slide()
title_bar(s, "How the model is structured", "Original model")
cards = [
    ("run.py", "Single entry point the user runs; selects a dataset and config."),
    ("empire.py", "Abstract Pyomo formulation: sets, variables, constraints, "
     "objective, and result printing."),
    ("reader.py", "Builds .tab input files from the Excel workbooks in 'Data handler'."),
    ("scenario_random.py", "Samples representative time series into stochastic "
     "operational scenarios."),
    ("config.py", "Configuration objects driven by config/run.yaml."),
    ("model_runner.py", "Wires config, data managers and the solve together."),
]
x0, y0 = Inches(0.55), Inches(1.6)
cw, ch = Inches(3.95), Inches(1.55)
gx, gy = Inches(0.2), Inches(0.25)
for i, (h, d) in enumerate(cards):
    r, c = divmod(i, 3)
    x = x0 + c * (cw + gx)
    y = y0 + r * (ch + gy)
    card = rect(s, x, y, cw, ch, LIGHT)
    rect(s, x, y, Inches(0.10), ch, ACCENT)
    tb = textbox(s, x + Inches(0.25), y + Inches(0.12), cw - Inches(0.4), ch - Inches(0.2))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    set_run(p.add_run(), h, 16, DARK, bold=True, font="Consolas")
    p2 = tf.add_paragraph()
    p2.space_before = Pt(4)
    set_run(p2.add_run(), d, 12.5, GREY)
tb = textbox(s, Inches(0.55), Inches(5.7), Inches(12.2), Inches(1.2))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Inputs: 7 Excel workbooks per dataset (general, generation, node, sets, "
        "transmission, storage) → .tab files → Pyomo model. "
        "Outputs: CSV results + optional IAMC-format export.", 14, GREY, italic=True)

# ======================================================================
# Slide 4 — Why changes were needed (motivation)
# ======================================================================
s = add_slide()
title_bar(s, "Motivation for the fork", "Why change it")
bullets(s, [
    "Goal: model the North Sea region in detail — countries NO, DK, UK, NL and DE, "
    "with explicit offshore (wind) nodes and the offshore grid linking them.",
    "Base EMPIRE had no proper handling of offshore nodes — it crashed on datasets "
    "where offshore nodes were absent or empty.",
    "IAMC result export produced incorrect / incomplete output for the North Sea "
    "nodes and technologies.",
    "Large North Sea runs solved very slowly in Gurobi, motivating dedicated solver "
    "performance work.",
    "Need: handle offshore nodes robustly, diagnose infeasibilities, and accelerate "
    "the solve without changing the physics.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.5), size=17, gap=11)

# ======================================================================
# Slide 5 — Changes overview (themes)
# ======================================================================
s = add_slide()
title_bar(s, "Overview of my changes", "What I did")
tb = textbox(s, Inches(0.55), Inches(1.3), Inches(12.2), Inches(0.5))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "18 commits ahead of upstream — ~370 lines of core model code; the rest is "
        "North Sea data, analysis notebooks and tooling.", 14, GREY, italic=True)
themes = [
    ("1  North Sea case", "NO/DK/UK/NL/DE with offshore nodes; safe loading "
     "when a dataset has no offshore nodes.", ACCENT),
    ("2  IAMC reporting", "Correct region & generator mappings; optional offshore→"
     "onshore aggregation.", ACCENT2),
    ("3  Solver performance", "Configurable Gurobi options, emission-cap "
     "reformulation, ramping toggle.", ACCENT),
    ("4  Robustness", "Infeasibility detection + IIS export; fixed CO2-price output bug.", ACCENT2),
    ("5  Tooling & env", "Result-analysis scripts/notebooks; pinned Python 3.11 env.", GREY),
]
y = Inches(1.95)
for h, d, col in themes:
    rect(s, Inches(0.55), y, Inches(12.23), Inches(0.92), LIGHT)
    rect(s, Inches(0.55), y, Inches(0.14), Inches(0.92), col)
    tb = textbox(s, Inches(0.9), y + Inches(0.06), Inches(3.4), Inches(0.8))
    set_run(tb.text_frame.paragraphs[0].add_run(), h, 17, DARK, bold=True)
    tb = textbox(s, Inches(4.3), y + Inches(0.06), Inches(8.3), Inches(0.8))
    tb.text_frame.word_wrap = True
    set_run(tb.text_frame.paragraphs[0].add_run(), d, 14, GREY)
    y += Inches(1.02)

# ======================================================================
# Slide 6 — Theme 1: North Sea region
# ======================================================================
s = add_slide()
title_bar(s, "1 — North Sea region & offshore nodes", "Change detail")
bullets(s, [
    "Added a North Sea case covering NO, DK, UK, NL and DE, with explicit offshore "
    "(wind) nodes representing the offshore grid.",
    "Added a north_sea mode and offshore-node handling in empire.py "
    "(new OffshoreNode set).",
    ("Sets_OffshoreNode.tab is now loaded only if it exists and is non-empty — fixes "
     "a crash on datasets with no offshore nodes.", 1),
    "New North Sea input dataset under Data handler/north_sea.",
    "Reader / tab-file generation updated to emit offshore sets correctly.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.0), size=17, gap=10)
tb = textbox(s, Inches(0.55), Inches(6.2), Inches(12.2), Inches(0.8))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Files: empire/core/empire.py, empire/core/reader.py, Data handler/north_sea",
        12.5, ACCENT, italic=True, font="Consolas")

# ======================================================================
# Slide 7 — Theme 2: IAMC reporting
# ======================================================================
s = add_slide()
title_bar(s, "2 — IAMC output corrections", "Change detail")
bullets(s, [
    "IAMC = standard reporting format for integrated-assessment / energy scenarios; "
    "EMPIRE can export results in it.",
    "Fixed region mapping for the new North Sea nodes, with a fallback for unknown "
    "nodes instead of failing.",
    "Fixed generator-name mapping: added spaced-name variants "
    "(\"Wind offshore floating/grounded\" → Wind|Offshore) and missing LigniteCCS / BioCCS.",
    "Collapsed technology-specific emission rows into the aggregate electricity-CO2 row.",
    "New optional flag aggregate_offshore_nodes_in_IAMC: maps each offshore node to its "
    "nearest onshore region (by coordinates) in the IAMC export.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.0), size=16.5, gap=9)
tb = textbox(s, Inches(0.55), Inches(6.3), Inches(12.2), Inches(0.8))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Files: empire/core/empire.py (IAMC block), config/run.yaml, config.py", 12.5,
        ACCENT, italic=True, font="Consolas")

# ======================================================================
# Slide 8 — Theme 3: Solver performance (headline)
# ======================================================================
s = add_slide()
title_bar(s, "3 — Solver performance acceleration", "Change detail")
bullets(s, [
    "Configurable Gurobi options plumbed config.py → model_runner.py → empire.py: "
    "Method, Crossover, Presolve, Threads, ScaleFlag, NumericFocus, BarHomogeneous.",
    ("Defaults enable the accelerators (barrier, crossover off, aggressive presolve); "
     "robustness 'brakes' are off by default and documented as enable-on-failure only.", 1),
    "Emission-cap reformulation: replaced one dense all-generators constraint row "
    "(which injected 5e-08 coefficients via /1e6) with per-node nodeEmission variables "
    "in tonnes — kills barrier fill-in and the bad matrix range.",
    ("Dual / CO2-price reporting updated to match the new formulation.", 1),
    "use_ramping toggle: optionally drop thermal inter-hour ramp constraints to reduce "
    "temporal coupling.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.5), size=16, gap=9)

# ======================================================================
# Slide 8b — Configurable Gurobi solver options (code)
# ======================================================================
s = add_slide()
title_bar(s, "3a — Configurable Gurobi solver options", "Change detail · code")
tb = textbox(s, Inches(0.55), Inches(1.3), Inches(12.2), Inches(0.7))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Solver tuning was hard-coded. It is now exposed end-to-end so each run can be "
        "tuned from config/run.yaml without touching the model:", 14, GREY, italic=True)
# flow chips
chips = ["config/run.yaml", "config.py", "model_runner.py", "empire.py → Gurobi"]
x = Inches(0.55)
for i, c in enumerate(chips):
    w = Inches(2.7)
    rect(s, x, Inches(2.05), w, Inches(0.5), ACCENT if i % 2 == 0 else DARK)
    tbc = textbox(s, x, Inches(2.12), w, Inches(0.4))
    pp = tbc.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
    set_run(pp.add_run(), c, 12, WHITE, bold=True, font="Consolas")
    if i < len(chips) - 1:
        ar = textbox(s, x + w, Inches(2.08), Inches(0.45), Inches(0.45))
        set_run(ar.text_frame.paragraphs[0].add_run(), "→", 20, ACCENT2, bold=True)
    x += w + Inches(0.45)
# before / after code
code_box(s, Inches(0.55), Inches(2.95), Inches(5.9), Inches(1.5),
         ["opt.options[\"Crossover\"] = 0",
          "opt.options[\"Method\"]    = 2"],
         title="Before — hard-coded", size=12)
code_box(s, Inches(6.85), Inches(2.95), Inches(5.9), Inches(3.4),
         ["opt.options[\"Method\"] = solver_method",
          "if solver_crossover is not None:",
          "    opt.options[\"Crossover\"]   = solver_crossover",
          "if solver_presolve is not None:",
          "    opt.options[\"Presolve\"]    = solver_presolve",
          "if solver_threads is not None:",
          "    opt.options[\"Threads\"]     = solver_threads",
          "if solver_scaleflag is not None:",
          "    opt.options[\"ScaleFlag\"]   = solver_scaleflag",
          "if solver_numericfocus is not None:",
          "    opt.options[\"NumericFocus\"]= solver_numericfocus",
          "if solver_barhomogeneous is not None:",
          "    opt.options[\"BarHomogeneous\"]= solver_barhomogeneous"],
         title="After — configurable (config-driven)", size=11)
bullets(s, [
    "Accelerators (defaults): Method=2 (barrier), Crossover=0 (skip the tail), "
    "Presolve=2 (aggressive).",
    "Brakes (off by default): ScaleFlag, NumericFocus, BarHomogeneous — robustness "
    "options that slow Gurobi; enable only on numerical failure.",
], Inches(0.55), Inches(4.55), Inches(6.0), Inches(2.6), size=12.5, gap=7)

# ======================================================================
# Slide 8c — Emission cap: original formulation (math)
# ======================================================================
s = add_slide()
title_bar(s, "3b — Emission cap constraint (original)", "Change detail · code")
tb = textbox(s, Inches(0.55), Inches(1.3), Inches(12.2), Inches(0.6))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "One constraint per period i and scenario w — a single sum over every "
        "generator and every operational hour, scaled by 1/1 000 000 (Mt):",
        14, GREY, italic=True)
code_box(s, Inches(0.55), Inches(1.95), Inches(12.23), Inches(2.55),
         ["def emission_cap_rule(model, i, w):",
          "    return sum(model.seasScale[s] * model.genCO2TypeFactor[g]",
          "               * (3.6 / model.genEfficiency[g,i]) * model.genOperational[n,g,h,i,w]",
          "               for (n,g) in model.GeneratorsOfNode",
          "               for (s,h) in model.HoursOfSeason) / 1000000 \\",
          "        - model.CO2cap[i] <= 0",
          "model.emission_cap = Constraint(model.PeriodActive, model.Scenario,",
          "                                rule=emission_cap_rule)"],
         title="empire.py — original", size=11.5)
bullets(s, [
    "Dense row: one constraint touches every (generator × hour) variable → hundreds "
    "of thousands of nonzeros in a single row → severe barrier-factorization fill-in.",
    "Bad scaling: the / 1000000 (Mt) factor puts ~5e-08 coefficients in the matrix, "
    "widening the range to [5e-08, 1e+01] and triggering Gurobi conditioning warnings.",
], Inches(0.55), Inches(4.75), Inches(12.2), Inches(2.4), size=15, gap=10)

# ======================================================================
# Slide 8d — Emission cap: reformulation (math + code)
# ======================================================================
s = add_slide()
title_bar(s, "3c — Emission cap reformulation", "Change detail · code")
tb = textbox(s, Inches(0.55), Inches(1.25), Inches(12.2), Inches(0.5))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Add a per-node emission variable (tonnes): split the dense row into sparse "
        "per-node rows + a small cap row, and move 1e6 to the right-hand side:",
        13.5, GREY, italic=True)
code_box(s, Inches(0.55), Inches(1.8), Inches(12.23), Inches(3.85),
         ["model.nodeEmission = Var(model.Node, model.PeriodActive, model.Scenario,",
          "                         domain=Reals)",
          "",
          ("def node_emission_rule(model, n, i, w):        # sparse: one node's generators",
           RGBColor(0xE6, 0xE9, 0xEF)),
          "    return sum(model.seasScale[s] * model.genCO2TypeFactor[g]",
          "               * (3.6 / model.genEfficiency[g,i]) * model.genOperational[n,g,h,i,w]",
          "               for g in model.Generator if (n,g) in model.GeneratorsOfNode",
          "               for (s,h) in model.HoursOfSeason) \\",
          "        - model.nodeEmission[n,i,w] == 0",
          "model.node_emission = Constraint(model.Node, model.PeriodActive,",
          "                                 model.Scenario, rule=node_emission_rule)",
          "",
          ("def emission_cap_rule(model, i, w):            # sparse: |N| terms, 1e6 on RHS",
           RGBColor(0xE6, 0xE9, 0xEF)),
          "    return sum(model.nodeEmission[n,i,w] for n in model.Node) \\",
          "        - 1e6 * model.CO2cap[i] <= 0",
          "model.emission_cap = Constraint(model.PeriodActive, model.Scenario,",
          "                                rule=emission_cap_rule)"],
         title="empire.py — reformulated", size=10.5)
bullets(s, [
    "Each definition row is sparse (only that node's generators); the cap row has just "
    "|N| terms — no dense row, no barrier fill-in. Moving 1e6 to the RHS removes the "
    "5e-08 coefficients → matrix range back to [1e-03, 1e+01].",
    "domain=Reals (CCS gives negative CO₂; cap can go negative). Dual / CO₂-price "
    "reporting updated to match (the *1e6 divisor removed).",
], Inches(0.55), Inches(5.75), Inches(12.2), Inches(1.6), size=12.5, gap=6)

# ======================================================================
# Slide 9 — Theme 3 evidence (performance findings)
# ======================================================================
s = add_slide()
title_bar(s, "3 — What the performance analysis found", "Change detail")
# two columns
colw = Inches(6.0)
tb = textbox(s, Inches(0.55), Inches(1.5), colw, Inches(5.0))
tf = tb.text_frame
p = tf.paragraphs[0]
set_run(p.add_run(), "Diagnosis", 18, ACCENT, bold=True)
for t in [
    "Barrier solve is fast; the crossover tail dominated — 9–10× the barrier time "
    "(millions of pivots).",
    "Wide matrix range [5e-08, …] came from the /1e6 in the emission cap, not from "
    "the input data (the tiny-value hypothesis was ruled out).",
    "One dense emission-cap row per (period, scenario) → catastrophic fill-in in the "
    "barrier Cholesky factor.",
    "Stacking NumericFocus / BarHomogeneous added 'brakes' that slowed large runs.",
]:
    pp = tf.add_paragraph(); pp.space_after = Pt(7)
    set_run(pp.add_run(), "•  " + t, 13.5, GREY)
tb = textbox(s, Inches(6.9), Inches(1.5), colw, Inches(5.0))
tf = tb.text_frame
p = tf.paragraphs[0]
set_run(p.add_run(), "Fixes applied", 18, ACCENT2, bold=True)
for t in [
    "Crossover = 0 → removes the 9–10× tail (interior solution).",
    "Method = 2 (barrier) + aggressive presolve.",
    "Emission cap expressed in tonnes → removes 5e-08 and the dense row.",
    "Brakes reverted to defaults; documented to enable only on numerical failure.",
    "All reasoning captured in PERF_HANDOFF.md (incl. a stale editable-install gotcha "
    "on the cluster).",
]:
    pp = tf.add_paragraph(); pp.space_after = Pt(7)
    set_run(pp.add_run(), "•  " + t, 13.5, GREY)

# ======================================================================
# Slide 10 — Theme 4: Robustness
# ======================================================================
s = add_slide()
title_bar(s, "4 — Robustness & diagnostics", "Change detail")
bullets(s, [
    "Infeasibility handling: on an 'infeasible' termination the model now writes an "
    ".lp file and computes an IIS (Irreducible Infeasible Subsystem).",
    ("Uses the gurobipy API with a gurobi_cl subprocess fallback — so a failed run "
     "produces a diagnosable file instead of just crashing.", 1),
    "Feasibility check added to prevent the model from crashing mid-build.",
    "Bug fix: results_co2_price_resolved.csv was never written (missing final row "
    "write) — now populated and includes a CO2Cap_Ton column.",
    "Node-name fixes for the North Sea dataset.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.0), size=17, gap=10)

# ======================================================================
# Slide 11 — Theme 5: Tooling & environment
# ======================================================================
s = add_slide()
title_bar(s, "5 — Tooling & environment", "Change detail")
bullets(s, [
    "New result-analysis tooling: scripts/plot.py, scripts/review_results.py, and "
    "notebooks results_reviewer.ipynb & scenarios_comparison.ipynb.",
    "IAMC-fixing helpers (check_iamc_mappings.py, _fix_iamc_excel.py, fix_node_names.py) "
    "and a regions geojson for mapping.",
    "environment.yml: pinned python=3.11 and dropped the 'defaults' conda channel for "
    "reproducible setup.",
    "config/run.yaml: new options surfaced (north_sea, aggregate_offshore_nodes_in_IAMC, "
    "use_ramping, solver_* block) with inline guidance.",
], Inches(0.55), Inches(1.55), Inches(12.2), Inches(5.0), size=17, gap=11)

# ======================================================================
# Slide 12 — Summary / impact
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(1.15), SW, Inches(0.06), ACCENT)
tb = textbox(s, Inches(0.55), Inches(0.3), Inches(12), Inches(0.9))
set_run(tb.text_frame.paragraphs[0].add_run(), "Summary & impact", 30, WHITE, bold=True)
pts = [
    "Extended EMPIRE to a North Sea model with explicit offshore nodes.",
    "Made the model robust: no crash without offshore nodes; infeasibilities are now "
    "diagnosable via IIS.",
    "Corrected IAMC reporting for the new regions and technologies.",
    "Accelerated the solve by attacking the real bottlenecks — crossover and a dense, "
    "badly-scaled emission-cap row — not just flag-tweaking.",
    "Added analysis tooling and a reproducible environment for ongoing scenario work.",
]
tb = textbox(s, Inches(0.7), Inches(1.6), Inches(12), Inches(5.2))
tf = tb.text_frame
for i, t in enumerate(pts):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(14)
    set_run(p.add_run(), "✓  " + t, 18, RGBColor(0xDE, 0xE5, 0xEF))
tb = textbox(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.6))
set_run(tb.text_frame.paragraphs[0].add_run(),
        "Physics unchanged where it matters — changes target resolution, reporting "
        "correctness, robustness and solver performance.", 13, ACCENT2, italic=True)

out = "OpenEMPIRE_changes_summary.pptx"
prs.save(out)
print("Saved", out, "with", len(prs.slides._sldIdLst), "slides")
