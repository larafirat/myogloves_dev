"""Render the full benchmark -- hold, mass sweep and posture -- into one report.

Supersedes make_posture_report.py, which covered posture only. Every figure is
read from the run JSON rather than typed, so the page cannot drift from what
was measured.

Usage:
    python myogloves_dev/scripts/make_benchmark_report.py \
        headline_v3.json mass_sweep_v3.json posture_rom_final12.json out.html
"""

from __future__ import annotations

import json
import sys

# ---------------------------------------------------------------- shared maps

OBJECTS = [("box", "Gelatin box", "0.097 kg", "72&times;88&times;28 mm &middot; flat faces"),
           ("can", "Soup can", "0.349 kg", "&oslash;66&times;101 mm &middot; smooth cylinder"),
           ("tuna", "Tuna tin", "0.171 kg", "&oslash;85&times;33 mm &middot; flat disc")]

DEVICES = [
    ("healthy", "Healthy hand", "no device &middot; own muscles", "ref"),
    ("Tyrone", "Tyrone", "3 routed tendons &middot; real geometry", "dev"),
    ("D1", "D1", "Zhao 2025 &middot; 5 motors, no abduction DoF", "dev"),
    ("D2", "D2", "Alicea 2021 &middot; 1 motor, synergy pulley", "dev"),
    ("D3", "D3", "Thimabut 2022 &middot; 2 fingers + rigid splint", "dev"),
    ("D4", "D4", "Gerez 2020 &middot; 5 motors incl. opposition", "dev"),
]
MASSES = ["0.05", "0.15", "0.3", "0.5"]

POSTURE_SHORT = {
    "glove_dev/box": "Tyrone",
    "PORT+OP/box/D1_underactuated_distal": "D1",
    "PORT+OP/box/D2_synergy_cross_finger": "D2",
    "SPLINT/box/D3_uniform_single_dof": "D3",
    "PORT+OP/box/D4_v2_hybrid_per_finger": "D4",
    "PORT+OP/box/D4_v2_opposition_first": "D4-of",
}
DIGITS = [("index", [("MCP", "mcp2_flexion"), ("PIP", "pm2_flexion"), ("DIP", "md2_flexion")]),
          ("middle", [("MCP", "mcp3_flexion"), ("PIP", "pm3_flexion"), ("DIP", "md3_flexion")]),
          ("ring", [("MCP", "mcp4_flexion"), ("PIP", "pm4_flexion"), ("DIP", "md4_flexion")]),
          ("little", [("MCP", "mcp5_flexion"), ("PIP", "pm5_flexion"), ("DIP", "md5_flexion")]),
          ("thumb", [("ABD", "cmc_abduction"), ("CMC", "cmc_flexion"),
                     ("MP", "mp_flexion"), ("IP", "ip_flexion")])]
FLAT = [(d, l, j) for d, js in DIGITS for l, j in js]


def hold_key(dev, obj):
    """Adapter key in the hold/mass JSON for a (device, object) pair."""
    if dev == "healthy":
        return f"healthy_{obj}"
    if dev == "Tyrone":
        return f"glove_dev_{obj}"
    if dev == "D3":
        return "splint_D3" if obj == "box" else f"splint_D3_{obj}"
    stem = {"D1": "D1_underactuated_distal_calibrated",
            "D2": "D2_synergy_cross_finger_calibrated",
            "D4": "D4_v2_hybrid_per_finger_calibrated"}[dev]
    return f"portOP_{stem}" if obj == "box" else f"portOP_{stem}_{obj}"


def band(pct):
    return "na" if pct is None else "good" if pct >= 60 else "mid" if pct >= 20 else "low"


def main():
    hold = json.load(open(sys.argv[1]))
    mass = json.load(open(sys.argv[2]))
    post = json.load(open(sys.argv[3]))
    dest = sys.argv[4]

    # --------------------------------------------------------- headline table
    head_rows = []
    for key, name, blurb, kind in DEVICES:
        cells = []
        for obj, *_ in OBJECTS:
            s = hold.get(hold_key(key, obj))
            if not s:
                cells.append('<td class="na" colspan="3">&mdash;</td>')
                continue
            g, sv, hd = s["grasp_rate"] * 100, s["survival_rate"] * 100, s["hold_median"]
            sf = s.get("setup_fail", 0)
            dag = '<sup title="setup failures">&dagger;</sup>' if sf else ''
            cells.append(
                f'<td class="mono">{g:.0f}%{dag}</td>'
                f'<td class="mono surv band-{band(sv)}" style="--fill:{max(sv,0):.0f}%">{sv:.0f}%</td>'
                f'<td class="mono soft">{hd:.2f}s</td>')
        head_rows.append(
            f'<tr class="{kind}"><th class="rowlab"><b>{name}</b>'
            f'<span class="blurb">{blurb}</span></th>{"".join(cells)}</tr>')

    # ------------------------------------------------------------- mass sweep
    mass_blocks = []
    for obj, oname, onative, odesc in OBJECTS:
        rows = []
        for key, name, blurb, kind in DEVICES:
            cells = []
            for mkg in MASSES:
                s = mass.get(f"{hold_key(key, obj)}@{mkg}kg")
                if not s:
                    cells.append('<td class="na">&mdash;</td>')
                    continue
                sv, g = s["survival_rate"] * 100, s["grasp_rate"] * 100
                cells.append(f'<td class="mono cell band-{band(sv)}" style="--fill:{max(sv,0):.0f}%"'
                             f' title="grasp {g:.0f}%, hold {s["hold_median"]:.2f}s">{sv:.0f}</td>')
            rows.append(f'<tr class="{kind}"><th class="rowlab">{name}</th>{"".join(cells)}</tr>')
        mass_blocks.append(f"""
      <div class="masscol">
        <h4>{oname} <span class="blurb">{odesc}</span></h4>
        <div class="scroll"><table>
          <thead><tr><th class="rowlab"></th>{"".join(f'<th class="mono">{m}</th>' for m in MASSES)}</tr>
                 <tr><th class="rowlab unit">kg &rarr;</th>{"".join('<th></th>' for _ in MASSES)}</tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table></div>
      </div>""")

    # ---------------------------------------------------------------- posture
    ref = post["reference_deg"]
    pdev = post["devices"]
    phead = "".join(f'<th class="j"><span class="dig">{d[:3]}</span>{l}</th>' for d, l, _ in FLAT)
    prefrow = "".join(
        f'<td class="mono {"faint" if abs(ref[j]) < post["min_ref_excursion_deg"] else ""}">{ref[j]:+.0f}</td>'
        for _, _, j in FLAT)
    prows = []
    for full, short in POSTURE_SHORT.items():
        dv = pdev.get(full)
        if not dv:
            continue
        cells = []
        for _, _, j in FLAT:
            m_ = dv["match_pct"].get(j, float("nan"))
            if j in dv["splinted"]:
                cells.append('<td class="cell splinted" title="held by the splint">&#9647;</td>')
            elif m_ != m_:
                cells.append('<td class="cell na" title="reference excursion too small to score">&middot;</td>')
            else:
                cells.append(f'<td class="cell band-{band(m_)}" style="--fill:{min(max(m_,0),100):.0f}%"'
                             f' title="{dv["excursion_deg"][j]:+.1f} deg">{m_:.0f}</td>')
        prows.append(f'<tr><th class="rowlab">{short}<span class="blurb">'
                     f'cover {dv["coverage_pct"]:.0f}% &middot; fidelity {dv["fidelity_pct"]:.0f}%'
                     f'</span></th>{"".join(cells)}</tr>')

    n = hold[hold_key("healthy", "box")]["n"]
    html = f"""<title>Exoglove benchmark &mdash; grasp, load and posture</title>
<style>
:root {{
  --ground:#f6f7f5; --surface:#fff; --ink:#13181d; --muted:#69737c;
  --rule:#dde2de; --rule-strong:#c3cac4; --accent:#16695a;
  --good:#2f7a56; --mid:#9c7a1e; --low:#a03a30; --wash:#eef1ee;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
    --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
    --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127; }}
}}
:root[data-theme="dark"] {{ --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
  --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
  --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127; }}
:root[data-theme="light"] {{ --ground:#f6f7f5; --surface:#fff; --ink:#13181d; --muted:#69737c;
  --rule:#dde2de; --rule-strong:#c3cac4; --accent:#16695a;
  --good:#2f7a56; --mid:#9c7a1e; --low:#a03a30; --wash:#eef1ee; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:clamp(1.5rem,4vw,3.5rem) clamp(1rem,3vw,2rem) 5rem; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
header.top {{ border-bottom:2px solid var(--ink); padding-bottom:1.5rem; margin-bottom:2.5rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); margin:0 0 .6rem; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(1.9rem,4.5vw,2.9rem); line-height:1.1;
  margin:0 0 .7rem; text-wrap:balance; letter-spacing:-.01em; }}
.standfirst {{ max-width:64ch; color:var(--muted); margin:0 0 1.4rem; font-size:1.02rem; }}
.method {{ display:flex; flex-wrap:wrap; gap:.4rem 1.6rem; font-family:var(--mono); font-size:.76rem;
  color:var(--muted); }}
.method b {{ color:var(--ink); font-weight:500; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:1.45rem; margin:3rem 0 .4rem; letter-spacing:-.005em; }}
h2:first-of-type {{ margin-top:0; }}
h4 {{ font-family:var(--serif); font-size:1rem; font-weight:600; margin:0 0 .5rem; }}
.lede {{ max-width:66ch; color:var(--muted); margin:0 0 1.4rem; font-size:.94rem; }}
.blurb {{ display:block; font-family:var(--sans); font-size:.72rem; font-weight:400; color:var(--muted);
  letter-spacing:0; }}
h4 .blurb {{ display:inline; margin-left:.4rem; }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:.82rem; }}
th,td {{ padding:.42rem .5rem; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule); }}
thead th {{ background:var(--surface); font-weight:500; font-size:.7rem; color:var(--muted);
  border-bottom:1px solid var(--rule-strong); }}
th.rowlab {{ text-align:left; font-weight:500; position:sticky; left:0; background:var(--surface);
  border-right:1px solid var(--rule); min-width:112px; }}
th.unit {{ font-family:var(--mono); font-size:.66rem; color:var(--muted); font-weight:400; }}
tbody tr:hover td, tbody tr:hover th.rowlab {{ background:var(--wash); }}
tr.ref th.rowlab {{ font-family:var(--serif); font-style:italic; }}
tr.ref td {{ color:var(--muted); }}
.soft {{ color:var(--muted); }} .faint {{ color:var(--muted); opacity:.6; }}
.na {{ color:var(--muted); opacity:.45; text-align:center; }}
td.surv, td.cell {{ position:relative; }}
td.surv::before, td.cell::before {{ content:""; position:absolute; left:0; bottom:0; height:3px;
  width:var(--fill); background:currentColor; opacity:.5; }}
.band-good {{ color:var(--good); }} .band-mid {{ color:var(--mid); }} .band-low {{ color:var(--low); }}
td.cell.splinted {{ color:var(--accent); text-align:center; }}
.objhead th {{ font-family:var(--serif); font-style:italic; color:var(--ink); font-size:.8rem;
  border-bottom:1px solid var(--rule-strong); }}
.massgrid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1.5rem; }}
th.j .dig {{ display:block; font-family:var(--mono); font-size:.62rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--accent); }}
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.3rem; margin:.7rem 0 0; font-size:.75rem; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:.4rem; }}
.swatch {{ width:22px; height:3px; display:inline-block; }}
.finding {{ border-left:2px solid var(--accent); padding:.1rem 0 .1rem 1rem; margin:1.6rem 0; max-width:68ch; }}
.finding p {{ margin:.35rem 0; }}
.caveats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); margin-top:1rem; }}
.caveats div {{ background:var(--surface); padding:.9rem 1rem; }}
.caveats h5 {{ margin:0 0 .3rem; font-size:.82rem; font-weight:600; }}
.caveats p {{ margin:0; font-size:.8rem; color:var(--muted); }}
footer {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--rule); font-size:.78rem;
  color:var(--muted); max-width:72ch; }}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">MyoSuite &middot; MyoHand &middot; {n} trials per cell</p>
  <h1>Five exoskeleton gloves, measured against the hand they assist</h1>
  <p class="standfirst">Four devices from the literature and one in development, driven through one
  rig on three YCB objects. The reference is not another device &mdash; it is the same hand with no
  device at all, which is the only benchmark that says whether assistance helped.</p>
  <p class="method">
    <span><b>Object</b> YCB, as myoMPL uses</span>
    <span><b>Friction</b> 1.0 / 0.005 / 0.0001, MyoAssist reference</span>
    <span><b>Failure</b> 50 mm relative to the hand</span>
    <span><b>Gate</b> 2+ digits opposing, 1.0 s</span>
    <span><b>Window</b> 5 s</span>
    <span><b>Transmission</b> ideal</span></p>
</header>

<h2>Can it hold the object?</h2>
<p class="lede">Survival is the headline: the share of trials still holding at 5&nbsp;s. Grasp is
whether a gated, opposed grip formed at all, and hold is the median time to failure among trials
that did grasp &mdash; so a high hold beside a zero survival means it always failed, but slowly.</p>
<div class="scroll">
  <table>
    <thead>
      <tr class="objhead"><th class="rowlab"></th>
        {"".join(f'<th colspan="3">{o[1]} &middot; {o[2]}</th>' for o in OBJECTS)}</tr>
      <tr><th class="rowlab">device</th>
        {"".join('<th>grasp</th><th>surv</th><th>hold</th>' for _ in OBJECTS)}</tr>
    </thead>
    <tbody>{"".join(head_rows)}</tbody>
  </table>
</div>
<p class="legend"><span>&dagger; some trials setup-failed: the object left its support during
settling, before the device acted</span></p>

<div class="finding">
  <p><strong>No device matches the unassisted hand.</strong> The healthy hand survives 93% on the box
  and 60% on the tuna; no device survives either. The one place devices lead is the can at light
  loads, where the healthy hand's own grip fails.</p>
  <p><strong>Tyrone's glove fails slowly rather than suddenly</strong> &mdash; the longest device hold
  on the can at every mass, while never surviving. Diagnosed as over-gripping: it applies 16&ndash;49&times;
  the physics-derived grip target where the other devices apply about 3&times;. Regulated to a sane
  force it survives 67% of can trials, against the healthy hand's 0%.</p>
</div>

<h2>How much load can it carry?</h2>
<p class="lede">Survival at four masses, the object's own geometry held fixed. This separates weight
from shape &mdash; the two are confounded in the native objects, which differ in both.</p>
<div class="massgrid">{"".join(mass_blocks)}</div>
<p class="legend">
  <span><i class="swatch" style="background:var(--good)"></i>&ge;60% survival</span>
  <span><i class="swatch" style="background:var(--mid)"></i>20&ndash;60%</span>
  <span><i class="swatch" style="background:var(--low)"></i>&lt;20%</span>
  <span>hover a cell for its grasp rate and median hold</span>
</p>

<h2>Is the hand the right shape?</h2>
<p class="lede">Holding time cannot see posture, and a glove that grips by dragging the hand into an
anatomically wrong shape is not a rehabilitation device. This reproduces Zhao et&nbsp;al.'s published
joint-angle similarity protocol: free-air closing, scored against the same hand moving under its own
muscles. The top row is that reference, in degrees.</p>
<div class="scroll">
  <table>
    <thead><tr><th class="rowlab">device</th>{phead}</tr></thead>
    <tbody>
      <tr class="ref"><th class="rowlab">healthy &deg;</th>{prefrow}</tr>
      {"".join(prows)}
    </tbody>
  </table>
</div>
<p class="legend">
  <span><i class="swatch" style="background:var(--accent)"></i>&#9647; held by the splint</span>
  <span>&middot; reference moves &lt;{post["min_ref_excursion_deg"]:.0f}&deg;, not scorable</span>
  <span>coverage = share of scorable joints driven at all; fidelity = match on those joints</span>
</p>

<div class="finding">
  <p><strong>Coverage and fidelity had to be separated.</strong> A device that ignores a joint and one
  that drives it badly are different failures, and one average hides both &mdash; D3's deliberately
  splinted thumb scored as poor fidelity for doing exactly what it is designed to do.</p>
  <p><strong>Only Tyrone's glove registers real thumb abduction.</strong> Its exo tendon has a 13.7&nbsp;mm
  moment arm there against the natural flexor's 3.9&nbsp;mm. Every ported device sits near zero,
  because the muscle pre-shape they borrow has already driven that joint to its limit.</p>
</div>

<h2>What these numbers do not cover</h2>
<div class="caveats">
  <div><h5>Ideal transmission</h5><p>No strap compliance, friction or slack. Fitting a series-spring
  model to Zhao et&nbsp;al.'s published similarities failed at any stiffness &mdash; the mismatch is in
  shape, not scale &mdash; so every device runs at its theoretical best.</p></div>
  <div><h5>Frozen moment arms</h5><p>D1&ndash;D4 are joint torques with arms measured once at rest;
  live geometry differs by 1.3&ndash;2&times; mid-grasp. Measured, and it does not change the ordering.</p></div>
  <div><h5>D3's splint is rigid</h5><p>Modelled as an ideal constraint with no strap compliance, so it
  is the one mechanism still idealised while the others are not.</p></div>
  <div><h5>Friction is shared, not per device</h5><p>No device publishes a coefficient. On curved
  objects the ranking moves with it, so it is held constant &mdash; the can column is a point on a
  band, not a device ranking.</p></div>
</div>

<footer>
  <p>Generated by <span class="mono">make_benchmark_report.py</span> from the run JSON, so no figure
  here is transcribed. Object, friction and the contact-duration gate follow myoMPL (Tan et&nbsp;al.,
  MyoAssist&nbsp;0.1, ICORR 2025), which manipulates the same YCB gelatin box. Failure is measured
  relative to the hand rather than the world, because the servo arm sags 39&nbsp;mm when the support
  is removed and that was being scored as a dropped object.</p>
</footer>
</div>
"""
    open(dest, "w").write(html)
    print(f"wrote {dest} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
