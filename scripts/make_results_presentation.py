"""Build a PowerPoint comparing OpenEMPIRE North Sea results for two scenarios:
GoRES and REPowerEU++. Embeds plots produced by results_reviewer.ipynb and
adds headline-metric tables extracted from the result CSVs.
"""
import json
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- palette ----
DARK   = RGBColor(0x1F, 0x2A, 0x44)
ACCENT = RGBColor(0x1B, 0x8A, 0x8F)   # teal  -> GoRES
ACCENT2 = RGBColor(0xE8, 0x7A, 0x2B)  # orange -> REPowerEU++
LIGHT  = RGBColor(0xF2, 0xF4, 0xF7)
GREY   = RGBColor(0x55, 0x5B, 0x66)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GREEN  = RGBColor(0x1A, 0x98, 0x50)
RED    = RGBColor(0xD7, 0x30, 0x27)

SCEN = ["GoRES", "REPowerEU++"]
SCEN_COLOR = {"GoRES": ACCENT, "REPowerEU++": ACCENT2}
DS = {"GoRES": "dataset_NS_GoRES", "REPowerEU++": "dataset_NS_REPowerEU++"}

HERE = Path(__file__).resolve().parent
RESBASE = HERE / ".." / "Results" / "basic_run"
PLOTS = {s: RESBASE / DS[s] / "plots" for s in SCEN}
METRICS = json.loads((HERE / "_metrics.json").read_text(encoding="utf-8"))

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def shift_period(label, by=5):
    return re.sub(r"\d{4}", lambda m: str(int(m.group()) + by), label)


def add_slide():
    return prs.slides.add_slide(BLANK)


def rect(slide, x, y, w, h, color, line=None):
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
    rect(slide, 0, 0, SW, Inches(1.05), DARK)
    rect(slide, 0, Inches(1.05), SW, Inches(0.06), ACCENT)
    tb = textbox(slide, Inches(0.55), Inches(0.12), Inches(12.2), Inches(0.9))
    tf = tb.text_frame
    if kicker:
        p = tf.paragraphs[0]
        set_run(p.add_run(), kicker.upper(), 11, ACCENT2, bold=True)
        p2 = tf.add_paragraph()
        set_run(p2.add_run(), title, 26, WHITE, bold=True)
    else:
        set_run(tf.paragraphs[0].add_run(), title, 28, WHITE, bold=True)


def bullets(slide, items, x, y, w, h, size=16, gap=6, color0=DARK):
    tb = textbox(slide, x, y, w, h)
    tf = tb.text_frame
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        text, level = (item if isinstance(item, tuple) else (item, 0))
        p.level = level
        bullet = "•  " if level == 0 else "–  "
        set_run(p.add_run(), bullet + text, size - level,
                color0 if level == 0 else GREY, bold=(level == 0))
    return tb


def fit_image(slide, path, bx, by, bw, bh):
    """Place an image fit (contain) inside the box, centred."""
    path = Path(path)
    if not path.exists():
        rect(slide, bx, by, bw, bh, LIGHT)
        t = textbox(slide, bx, by + bh // 2, bw, Inches(0.4))
        t.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        set_run(t.text_frame.paragraphs[0].add_run(),
                f"[missing: {path.name}]", 11, RED, italic=True)
        return
    iw, ih = Image.open(path).size
    scale = min(bw / iw, bh / ih)
    w = int(iw * scale)
    h = int(ih * scale)
    x = bx + (bw - w) // 2
    y = by + (bh - h) // 2
    slide.shapes.add_picture(str(path), x, y, width=Emu(w), height=Emu(h))


def scen_label(slide, text, x, y, w, color):
    bar = rect(slide, x, y, w, Inches(0.34), color)
    tb = textbox(slide, x, y + Inches(0.01), w, Inches(0.32))
    p = tb.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    set_run(p.add_run(), text, 14, WHITE, bold=True)


def two_up(slide, fname, caption=None, top=Inches(1.5)):
    """Two scenario images side by side, each with a coloured scenario label."""
    margin = Inches(0.35)
    gap = Inches(0.3)
    colw = (SW - 2 * margin - gap) // 2
    img_top = top + Inches(0.42)
    img_h = SH - img_top - Inches(0.45)
    for i, s in enumerate(SCEN):
        x = margin + i * (colw + gap)
        scen_label(slide, s, x, top, colw, SCEN_COLOR[s])
        fit_image(slide, PLOTS[s] / fname, x, img_top, colw, img_h)
    if caption:
        tb = textbox(slide, margin, SH - Inches(0.42), SW - 2 * margin, Inches(0.35))
        set_run(tb.text_frame.paragraphs[0].add_run(), caption, 11.5, GREY, italic=True)


def table(slide, rows, x, y, w, h, header_fill=DARK, col_widths=None,
          font=11, first_col_left=True):
    nrows, ncols = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nrows, ncols, x, y, w, h).table
    if col_widths:
        for ci, cw in enumerate(col_widths):
            gt.columns[ci].width = cw
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = gt.cell(ri, ci)
            cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
            cell.margin_left = Pt(6); cell.margin_right = Pt(6)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (ci == 0 and first_col_left) else PP_ALIGN.CENTER
            color = WHITE if ri == 0 else DARK
            bold = (ri == 0) or (ci == 0)
            set_run(p.add_run(), str(val), font, color, bold=bold)
            if ri == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if ri % 2 else LIGHT
    return gt


# ======================================================================
# Slide 1 — Title
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(4.5), SW, Inches(0.08), ACCENT)
rect(s, Inches(6.66), Inches(4.5), Inches(6.66), Inches(0.08), ACCENT2)
tb = textbox(s, Inches(0.8), Inches(1.7), Inches(11.7), Inches(2.6))
tf = tb.text_frame
set_run(tf.paragraphs[0].add_run(),
        "OpenEMPIRE — North Sea", 42, WHITE, bold=True)
p = tf.add_paragraph()
set_run(p.add_run(), "Scenario results: GoRES  vs  REPowerEU++", 26,
        RGBColor(0xC9, 0xD2, 0xE0), bold=True)
p = tf.add_paragraph()
set_run(p.add_run(),
        "Capacity expansion of the European power system to 2060 with explicit "
        "North Sea offshore detail — two decarbonisation pathways compared.",
        16, RGBColor(0xA9, 0xB4, 0xC4))
tb = textbox(s, Inches(0.8), Inches(4.75), Inches(11.7), Inches(1.0))
set_run(tb.text_frame.paragraphs[0].add_run(), "Ehsan Nokandi", 16, ACCENT2, bold=True)
p = tb.text_frame.add_paragraph()
set_run(p.add_run(),
        "Generated from results_reviewer.ipynb  ·  Results/basic_run", 12,
        RGBColor(0x8A, 0x95, 0xA8))

# ======================================================================
# Slide 2 — Scenarios & setup
# ======================================================================
s = add_slide()
title_bar(s, "The two scenarios & model setup", "Context")
colw = Inches(6.05)
# GoRES card
for i, (name, desc) in enumerate([
    ("GoRES", ["Ambitious, RES-driven decarbonisation pathway.",
               "Drives the system to deep net-negative CO₂ very early "
               "(emissions go negative by ~2030).",
               "Largest build-out: heavy solar + onshore wind, with BioCCS "
               "providing negative emissions."]),
    ("REPowerEU++", ["Pathway anchored on the EU REPowerEU energy strategy.",
                     "Slower, more gradual decarbonisation — emissions stay "
                     "positive into the early 2040s.",
                     "Smaller total build; more offshore wind & waste, less "
                     "solar and onshore wind than GoRES."]),
]):
    x = Inches(0.4) + i * (colw + Inches(0.4))
    rect(s, x, Inches(1.4), colw, Inches(0.5), SCEN_COLOR[name])
    tb = textbox(s, x, Inches(1.44), colw, Inches(0.45))
    tb.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    set_run(tb.text_frame.paragraphs[0].add_run(), name, 18, WHITE, bold=True)
    rect(s, x, Inches(1.9), colw, Inches(2.7), LIGHT)
    bullets(s, desc, x + Inches(0.2), Inches(2.05), colw - Inches(0.4),
            Inches(2.5), size=14, gap=8)
bullets(s, [
    "Model: OpenEMPIRE — two-stage stochastic capacity-expansion of the European "
    "power system, 128 nodes, with explicit North Sea offshore (wind) nodes.",
    "Horizon: investment periods 2025–2030 … 2055–2060; 3 stochastic "
    "operational scenarios; representative seasons + peak hours.",
    "Outputs reviewed: installed capacity, annual generation mix, CO₂ vs cap, "
    "transmission build-out and expected load curtailment.",
], Inches(0.4), Inches(4.75), Inches(12.5), Inches(2.4), size=14, gap=8)

# ======================================================================
# Slide 3 — Headline comparison table
# ======================================================================
s = add_slide()
title_bar(s, "Headline comparison", "Key numbers · final period 2055–2060")
g, r = METRICS["GoRES"], METRICS["REPowerEU++"]


def first_negative(co2):
    for k in sorted(co2):
        if co2[k] < 0:
            return shift_period(k)
    return "—"


rows = [
    ["Metric", "GoRES", "REPowerEU++"],
    ["Total discounted system cost", f"€{g['objective_BEUR']:,.0f} bn",
     f"€{r['objective_BEUR']:,.0f} bn"],
    ["Total installed capacity (2055–60)", f"{g['total_installed_GW']:,.0f} GW",
     f"{r['total_installed_GW']:,.0f} GW"],
    ["Total annual generation (2055–60)", f"{g['total_prod_TWh']:,.0f} TWh",
     f"{r['total_prod_TWh']:,.0f} TWh"],
    ["CO₂ emissions, final period", f"{g['co2_Mt'][g['last_period']]:,.0f} Mt/yr",
     f"{r['co2_Mt'][r['last_period']]:,.0f} Mt/yr"],
    ["First net-negative CO₂ period", first_negative(g["co2_Mt"]),
     first_negative(r["co2_Mt"])],
]
table(s, rows, Inches(1.4), Inches(1.7), Inches(10.5), Inches(3.4),
      col_widths=[Inches(5.0), Inches(2.75), Inches(2.75)], font=15)
tb = textbox(s, Inches(1.4), Inches(5.5), Inches(10.5), Inches(1.6))
bullets(s, [
    "GoRES decarbonises far harder and earlier — net-negative emissions already "
    "around 2030 — but at ~44% higher system cost and a much larger build.",
    "REPowerEU++ reaches a more modest end-state (−139 vs −254 Mt) with a "
    "smaller, cheaper system.",
], Inches(1.4), Inches(5.5), Inches(10.5), Inches(1.6), size=14, gap=8)

# ======================================================================
# Slide 4 — CO2 trajectory
# ======================================================================
s = add_slide()
title_bar(s, "CO₂ emissions vs cap", "Decarbonisation pace")
two_up(s, "co2_emissions_vs_cap.png",
       caption="Annual CO₂ emission vs the binding cap, per period and scenario. "
               "Negative values reflect net-negative emissions from CCS (BioCCS).")

# ======================================================================
# Slide 5 — Annual generation mix
# ======================================================================
s = add_slide()
title_bar(s, "Annual generation by technology", "Generation mix over time")
two_up(s, "annual_generation_stacked.png",
       caption="Stacked annual generation [GWh] by technology and investment period "
               "(Europe-wide, from results_output_EuropeSummary.csv).")

# ======================================================================
# Slide 6 — Total installed capacity stacked
# ======================================================================
s = add_slide()
title_bar(s, "Total installed capacity by technology", "Capacity build-out")
two_up(s, "total_installed_capacity_stacked.png",
       caption="Total installed capacity [MW] by technology and investment period.")

# ======================================================================
# Slide 7 — Installed capacity by tech (table)
# ======================================================================
s = add_slide()
title_bar(s, "Installed capacity by technology — 2055–2060", "Final-period mix [GW]")
techs = ["Solar", "Windonshore", "Windoffshoregrounded", "Windoffshorefloating",
         "Hydrorun-of-the-river", "Hydroregulated", "BioCCS", "GasCCGT", "GasOCGT",
         "Nuclear", "Wave", "Waste", "Coal", "Lignite"]
nice = {"Windonshore": "Wind onshore", "Windoffshoregrounded": "Wind offshore (grounded)",
        "Windoffshorefloating": "Wind offshore (floating)",
        "Hydrorun-of-the-river": "Hydro run-of-river", "Hydroregulated": "Hydro regulated"}
rows = [["Technology", "GoRES [GW]", "REPowerEU++ [GW]", "Δ (G−R)"]]
gi, ri = g["installed_GW_by_tech"], r["installed_GW_by_tech"]
for t in techs:
    gv, rv = gi.get(t, 0.0), ri.get(t, 0.0)
    if gv < 0.05 and rv < 0.05:
        continue
    rows.append([nice.get(t, t), f"{gv:,.1f}", f"{rv:,.1f}", f"{gv - rv:+,.1f}"])
rows.append(["TOTAL", f"{g['total_installed_GW']:,.0f}", f"{r['total_installed_GW']:,.0f}",
             f"{g['total_installed_GW'] - r['total_installed_GW']:+,.0f}"])
table(s, rows, Inches(2.0), Inches(1.35), Inches(9.3), Inches(5.8),
      col_widths=[Inches(3.6), Inches(1.9), Inches(2.1), Inches(1.7)], font=12.5)

# ======================================================================
# Slide 8 — Generation capacity map (final period)
# ======================================================================
s = add_slide()
title_bar(s, "Installed generation capacity by area — 2055–2060", "Spatial build-out")
two_up(s, "gen_capacity_2055-2060.png",
       caption="Installed-capacity pies per node in the final period; offshore nodes "
               "aggregated to nearest onshore region. Lines = transmission capacity.")

# ======================================================================
# Slides 9-10 — Wind & Solar capacity maps
# ======================================================================
for fname, title, kick in [
    ("cap_bars_wind.png", "Wind capacity by area & period", "Wind"),
    ("cap_bars_solar.png", "Solar capacity by area & period", "Solar"),
]:
    s = add_slide()
    title_bar(s, title, kick)
    two_up(s, fname,
           caption="One bar per node per investment period; bar height = installed "
                   "capacity [MW]; colour = period.")

# ======================================================================
# Slide 11 — Capacity change wind & solar
# ======================================================================
s = add_slide()
title_bar(s, "Capacity change — wind & solar (first → last period)", "Where growth lands")
two_up(s, "capacity_change_wind_and_solar.png",
       caption="Net change in wind + solar installed capacity by node, first to last "
               "investment period.")

# ======================================================================
# Slide 12 — Thermal / coal phase-out
# ======================================================================
s = add_slide()
title_bar(s, "Thermal & nuclear capacity change", "Fossil phase-down")
two_up(s, "capacity_change_thermal_generation_including_nuclear.png",
       caption="Net change in thermal + nuclear capacity by node, first → last period.")

# ======================================================================
# Slide 13 — Transmission final capacity
# ======================================================================
s = add_slide()
title_bar(s, "Transmission capacity — final period", "Grid build-out")
two_up(s, "transmission_capacity_final.png",
       caption="All transmission lines in the final period; width = installed capacity, "
               "colour = % change vs first period (red ↓, grey ≈, green ↑).")

# ======================================================================
# Slide 14 — Load curtailment
# ======================================================================
s = add_slide()
title_bar(s, "Expected load curtailment — final period", "Reliability")
two_up(s, "load_curtailment_2055-2060.png",
       caption="Expected load curtailment [GWh] by area in the final period; bubble "
               "size scales with curtailed energy.")

# ======================================================================
# Slide 15 — Key takeaways
# ======================================================================
s = add_slide()
rect(s, 0, 0, SW, SH, DARK)
rect(s, 0, Inches(1.05), SW, Inches(0.06), ACCENT)
tb = textbox(s, Inches(0.55), Inches(0.28), Inches(12), Inches(0.8))
set_run(tb.text_frame.paragraphs[0].add_run(), "Key takeaways", 30, WHITE, bold=True)
pts = [
    f"GoRES is the deep-decarbonisation pathway: net-negative CO₂ from ~2030, "
    f"ending at {g['co2_Mt'][g['last_period']]:,.0f} Mt/yr via heavy BioCCS, solar and "
    f"onshore wind.",
    f"REPowerEU++ decarbonises more gradually — positive emissions into the early "
    f"2040s, ending at {r['co2_Mt'][r['last_period']]:,.0f} Mt/yr.",
    f"GoRES costs ~{(g['objective_BEUR'] / r['objective_BEUR'] - 1) * 100:,.0f}% more "
    f"(€{g['objective_BEUR']:,.0f} bn vs €{r['objective_BEUR']:,.0f} bn) and "
    f"builds {g['total_installed_GW'] - r['total_installed_GW']:,.0f} GW more capacity.",
    "Both lean hard on solar + wind; GoRES adds far more solar and BioCCS, while "
    "REPowerEU++ leans relatively more on offshore wind and waste.",
    "North Sea offshore wind and the offshore grid expand in both pathways, "
    "reinforcing cross-border transmission between NO, DK, UK, NL and DE.",
]
tb = textbox(s, Inches(0.7), Inches(1.45), Inches(12), Inches(5.6))
tf = tb.text_frame
for i, t in enumerate(pts):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_after = Pt(14)
    set_run(p.add_run(), "✓  " + t, 17, RGBColor(0xDE, 0xE5, 0xEF))

out = HERE / ".." / "OpenEMPIRE_NS_scenario_results.pptx"
prs.save(str(out))
print("Saved", out.resolve(), "with", len(prs.slides._sldIdLst), "slides")
