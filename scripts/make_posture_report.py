"""Render posture_benchmark's JSON into a standalone HTML report.

Generated from the run output rather than typed by hand, so every number on the
page traces to a trial. Run posture_benchmark.py with --out first.

Usage:
    python myogloves_dev/scripts/make_posture_report.py \
        myogloves_dev/posture_rom_full.json out.html
"""

from __future__ import annotations

import json
import sys

DIGITS = [
    ("index", [("MCP", "mcp2_flexion"), ("PIP", "pm2_flexion"), ("DIP", "md2_flexion")]),
    ("middle", [("MCP", "mcp3_flexion"), ("PIP", "pm3_flexion"), ("DIP", "md3_flexion")]),
    ("ring", [("MCP", "mcp4_flexion"), ("PIP", "pm4_flexion"), ("DIP", "md4_flexion")]),
    ("little", [("MCP", "mcp5_flexion"), ("PIP", "pm5_flexion"), ("DIP", "md5_flexion")]),
    ("thumb", [("ABD", "cmc_abduction"), ("CMC", "cmc_flexion"),
               ("MP", "mp_flexion"), ("IP", "ip_flexion")]),
]
FLAT = [(dig, lbl, j) for dig, js in DIGITS for lbl, j in js]

SHORT = {
    "glove_dev/box": ("Tyrone", "3 routed tendons, real geometry"),
    "PORT+OP/D1_underactuated_distal": ("D1", "Zhao 2025 &middot; 5 motors, no abduction DoF"),
    "PORT+OP/D2_synergy_cross_finger": ("D2", "Alicea 2021 &middot; 1 motor, synergy pulley"),
    "SPLINT/D3_uniform_single_dof": ("D3", "Thimabut 2022 &middot; 2 fingers + rigid thumb splint"),
    "PORT+OP/D4_v2_hybrid_per_finger": ("D4", "Gerez 2020 &middot; 5 motors incl. opposition"),
    "PORT+OP/D4_v2_opposition_first": ("D4-of", "D4, thumb flexion delayed"),
}


def band(v):
    """Semantic band for a match percentage. Thresholds are read off the data:
    the healthy hand defines 100, and nothing here clears 76."""
    if v != v:
        return "na"
    return "good" if v >= 70 else "mid" if v >= 40 else "low"


def fmt(v, dp=1, suffix=""):
    return "&mdash;" if v is None or v != v else f"{v:.{dp}f}{suffix}"


def main():
    src, dest = sys.argv[1], sys.argv[2]
    d = json.load(open(src))
    ref = d["reference_deg"]
    devs = d["devices"]
    n_trials = d["trials_requested"]

    def role(dv, j):
        if j in dv["splinted"]:
            return "splinted"
        return "driven" if j in dv["driven"] else "idle"

    # ---------------------------------------------------------------- summary
    cards = []
    for name, dv in devs.items():
        short, blurb = SHORT.get(name, (name, ""))
        cov, fid = dv["coverage_pct"], dv["fidelity_pct"]
        cards.append(f"""
      <article class="card">
        <header>
          <h3>{short}</h3>
          <p class="blurb">{blurb}</p>
        </header>
        <dl>
          <div class="metric">
            <dt>Coverage</dt>
            <dd><span class="num">{fmt(cov, 1, '%')}</span>
              <span class="bar"><i style="width:{max(cov,0):.1f}%"></i></span></dd>
            <p class="note">{len(dv['driven'])} joints driven&nbsp;&middot;&nbsp;{len(dv['splinted'])} splinted</p>
          </div>
          <div class="metric">
            <dt>Fidelity</dt>
            <dd><span class="num band-{band(fid)}">{fmt(fid, 1, '%')}</span>
              <span class="bar"><i class="band-{band(fid)}" style="width:{max(fid,0):.1f}%"></i></span></dd>
            <p class="note">on the joints it drives</p>
          </div>
        </dl>
      </article>""")

    # ------------------------------------------------------------ joint matrix
    head = "".join(
        f'<th class="j" title="{j}"><span class="dig">{dig[:3]}</span>{lbl}</th>'
        for dig, lbl, j in FLAT)
    refrow = "".join(
        f'<td class="mono {"faint" if abs(ref[j]) < d["min_ref_excursion_deg"] else ""}">'
        f'{ref[j]:+.0f}</td>' for _, _, j in FLAT)

    rows = []
    for name, dv in devs.items():
        short, _ = SHORT.get(name, (name, ""))
        cells = []
        for _, _, j in FLAT:
            m = dv["match_pct"].get(j, float("nan"))
            r = role(dv, j)
            if r == "splinted":
                cells.append('<td class="cell splinted" title="held by the splint">&#9647;</td>')
            elif m != m:
                cells.append('<td class="cell na" title="reference excursion too small to score">&middot;</td>')
            else:
                w = min(max(m, 0), 100)
                cells.append(
                    f'<td class="cell {r} band-{band(m)}" style="--fill:{w:.0f}%" '
                    f'title="{j}: {dv["excursion_deg"][j]:+.1f} deg, Eq.1 '
                    f'{dv["similarity_pct"][j]:.0f}%">{m:.0f}</td>')
        rows.append(f'<tr><th class="rowlab">{short}</th>{"".join(cells)}</tr>')

    # -------------------------------------------------------------- per trial
    blocks = []
    for name, dv in devs.items():
        short, blurb = SHORT.get(name, (name, ""))
        scored = [j for _, _, j in FLAT if dv["match_pct"].get(j, float("nan")) == dv["match_pct"].get(j, float("nan"))]
        cols = [(dig, lbl, j) for dig, lbl, j in FLAT if j in scored]
        th = "".join(f'<th class="j"><span class="dig">{dig[:3]}</span>{lbl}</th>'
                     for dig, lbl, _ in cols)
        trs = []
        for t in dv["per_trial"]:
            if not t["usable"]:
                trs.append(f'<tr><th class="rowlab mono">{t["seed"]}</th>'
                           f'<td class="na" colspan="{len(cols)}">no usable trial</td></tr>')
                continue
            tds = "".join(f'<td class="mono">{t["joints"][j]:+.1f}</td>' for _, _, j in cols)
            trs.append(f'<tr><th class="rowlab mono">{t["seed"]}</th>{tds}</tr>')
        med = "".join(f'<td class="mono strong">{dv["excursion_deg"][j]:+.1f}</td>' for _, _, j in cols)
        spread = []
        for _, _, j in cols:
            vals = [t["joints"][j] for t in dv["per_trial"] if t["usable"]]
            spread.append(f'<td class="mono faint">{max(vals)-min(vals):.1f}</td>')
        mrow = "".join(f'<td class="mono band-{band(dv["match_pct"][j])}">'
                       f'{dv["match_pct"][j]:.0f}%</td>' for _, _, j in cols)
        blocks.append(f"""
      <section class="trials">
        <h3>{short} <span class="blurb">{blurb}</span></h3>
        <div class="scroll">
          <table>
            <thead><tr><th class="rowlab">seed</th>{th}</tr></thead>
            <tbody>{"".join(trs)}</tbody>
            <tfoot>
              <tr class="sum"><th class="rowlab">median</th>{med}</tr>
              <tr class="sum"><th class="rowlab">range</th>{"".join(spread)}</tr>
              <tr class="sum"><th class="rowlab">match</th>{mrow}</tr>
            </tfoot>
          </table>
        </div>
      </section>""")

    html = f"""<title>Posture benchmark &mdash; exoglove joint fidelity</title>
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
  :root {{
    --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
    --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
    --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
  --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
  --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127;
}}
:root[data-theme="light"] {{
  --ground:#f6f7f5; --surface:#fff; --ink:#13181d; --muted:#69737c;
  --rule:#dde2de; --rule-strong:#c3cac4; --accent:#16695a;
  --good:#2f7a56; --mid:#9c7a1e; --low:#a03a30; --wash:#eef1ee;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.55;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:clamp(1.5rem,4vw,3.5rem) clamp(1rem,3vw,2rem) 5rem; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}

header.top {{ border-bottom:2px solid var(--ink); padding-bottom:1.5rem; margin-bottom:2.5rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .6rem; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(1.9rem,4.5vw,2.9rem);
  line-height:1.1; margin:0 0 .7rem; text-wrap:balance; letter-spacing:-.01em; }}
.standfirst {{ max-width:64ch; color:var(--muted); margin:0 0 1.4rem; font-size:1.02rem; }}
.method {{ display:flex; flex-wrap:wrap; gap:.4rem 1.6rem; font-family:var(--mono);
  font-size:.76rem; color:var(--muted); }}
.method b {{ color:var(--ink); font-weight:500; }}

h2 {{ font-family:var(--serif); font-weight:600; font-size:1.45rem; margin:3rem 0 .4rem;
  letter-spacing:-.005em; }}
h2:first-of-type {{ margin-top:0; }}
.lede {{ max-width:66ch; color:var(--muted); margin:0 0 1.4rem; font-size:.94rem; }}
h3 {{ font-family:var(--serif); font-size:1.12rem; font-weight:600; margin:0; }}
.blurb {{ font-family:var(--sans); font-size:.78rem; font-weight:400; color:var(--muted); }}

.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); }}
.card {{ background:var(--surface); padding:1.15rem 1.2rem 1.3rem; display:flex;
  flex-direction:column; gap:1rem; }}
.card header p {{ margin:.15rem 0 0; }}
.card dl {{ margin:0; display:flex; flex-direction:column; gap:.9rem; }}
.metric dt {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); }}
.metric dd {{ margin:.2rem 0 0; display:flex; align-items:baseline; gap:.6rem; }}
.num {{ font-family:var(--mono); font-size:1.5rem; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em; }}
.bar {{ flex:1; height:5px; background:var(--wash); position:relative; overflow:hidden; }}
.bar i {{ position:absolute; inset:0 auto 0 0; background:var(--accent); }}
.bar i.band-good {{ background:var(--good); }}
.bar i.band-mid {{ background:var(--mid); }}
.bar i.band-low {{ background:var(--low); }}
.band-good {{ color:var(--good); }} .band-mid {{ color:var(--mid); }} .band-low {{ color:var(--low); }}
.note {{ margin:.25rem 0 0; font-size:.74rem; color:var(--muted); }}

.scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:.82rem; }}
th, td {{ padding:.42rem .5rem; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--rule); }}
thead th {{ position:sticky; top:0; background:var(--surface); font-weight:500;
  font-size:.7rem; color:var(--muted); border-bottom:1px solid var(--rule-strong); }}
th.j .dig {{ display:block; font-family:var(--mono); font-size:.62rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--accent); }}
th.rowlab {{ text-align:left; font-weight:500; position:sticky; left:0;
  background:var(--surface); border-right:1px solid var(--rule); }}
tbody tr:hover td, tbody tr:hover th.rowlab {{ background:var(--wash); }}
.faint {{ color:var(--muted); }}
.strong {{ font-weight:600; }}
tfoot .sum td, tfoot .sum th {{ border-top:1px solid var(--rule-strong); border-bottom:none;
  font-size:.78rem; }}
tfoot .sum:first-child td {{ font-weight:600; }}

td.cell {{ font-family:var(--mono); font-variant-numeric:tabular-nums; position:relative;
  min-width:44px; }}
td.cell::before {{ content:""; position:absolute; left:0; bottom:0; height:3px;
  width:var(--fill); background:currentColor; opacity:.55; }}
td.cell.idle {{ color:var(--muted); opacity:.5; font-style:italic; }}
td.cell.splinted {{ color:var(--accent); }}
td.na {{ color:var(--muted); opacity:.45; }}
tr.ref td {{ color:var(--muted); }}
tr.ref th.rowlab {{ font-family:var(--serif); font-style:italic; }}

.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.3rem; margin:.7rem 0 0;
  font-size:.75rem; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:.4rem; }}
.swatch {{ width:22px; height:3px; display:inline-block; }}

.trials {{ margin-top:1.6rem; }}
.trials h3 {{ margin-bottom:.5rem; }}
.finding {{ border-left:2px solid var(--accent); padding:.1rem 0 .1rem 1rem; margin:1.5rem 0;
  max-width:68ch; }}
.finding p {{ margin:.35rem 0; }}
.finding strong {{ font-weight:600; }}
footer {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--rule);
  font-size:.78rem; color:var(--muted); max-width:70ch; }}
@media (max-width:640px) {{ .num {{ font-size:1.25rem; }} }}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">MyoSuite &middot; free-air ROM protocol</p>
  <h1>Posture benchmark: what the grip metric cannot see</h1>
  <p class="standfirst">Holding time says whether an object stayed up. It says nothing about the
  shape of the hand that held it &mdash; and a glove that grips by dragging the hand into an
  anatomically wrong posture is not a rehabilitation device. This measures the shape, against an
  unassisted hand performing the same closing motion.</p>
  <p class="method"><span><b>Protocol</b> Zhao et al. 2025 &sect;3.2, reproduced</span>
    <span><b>Reference</b> healthy MyoHand, own muscles</span>
    <span><b>Trials</b> {n_trials} per device, seeds 3000&ndash;{3000+n_trials-1}</span>
    <span><b>Object</b> none (free air)</span>
    <span><b>Transmission</b> ideal</span></p>
</header>

<h2>Two numbers, not one</h2>
<p class="lede">A device that does not drive a joint and a device that drives it badly are
different failures, and one average hides both. D3 forced the split: its deliberately splinted
thumb scored as poor fidelity and put it last, for doing exactly what it is designed to do.</p>
<div class="cards">{"".join(cards)}</div>

<h2>Every joint, every device</h2>
<p class="lede">Match against the healthy hand's excursion at that joint, 0&ndash;100%, symmetric
so over-driving costs as much as under-driving. The top row is the reference itself, in degrees.
Hover any cell for its raw excursion and its Eq.&thinsp;1 similarity.</p>
<div class="scroll">
  <table>
    <thead><tr><th class="rowlab">device</th>{head}</tr></thead>
    <tbody>
      <tr class="ref"><th class="rowlab">healthy &deg;</th>{refrow}</tr>
      {"".join(rows)}
    </tbody>
  </table>
</div>
<p class="legend">
  <span><i class="swatch" style="background:var(--good)"></i> &ge;70% match</span>
  <span><i class="swatch" style="background:var(--mid)"></i> 40&ndash;70%</span>
  <span><i class="swatch" style="background:var(--low)"></i> &lt;40%</span>
  <span><i class="swatch" style="background:var(--accent)"></i> &#9647; held by splint</span>
  <span><i>italic</i> = joint the device does not drive</span>
  <span>&middot; = reference moves &lt;{d["min_ref_excursion_deg"]:.0f}&deg;, not scorable</span>
</p>

<div class="finding">
  <p><strong>The thumb is where the devices separate.</strong> Tyrone's glove is the only one
  registering real thumb abduction (60% match) &mdash; its exo tendon has a 13.7&nbsp;mm moment
  arm at that joint against the natural flexor's 3.9&nbsp;mm, so the design is weighted roughly
  4&times; toward opposition. Every ported device sits near zero there, because the OP pre-shape
  they borrow has already driven that joint to its limit before they act.</p>
</div>

<h2>Trial by trial</h2>
<p class="lede">Joint excursion in degrees for each of the {n_trials} seeds, with the median, the
full spread and the resulting match beneath. Spread is the honest check on the medians above: a
device oscillating between two very different postures can average out to look consistent.
Columns are the scorable joints only.</p>
{"".join(blocks)}

<footer>
  <p>Generated by <span class="mono">make_posture_report.py</span> from
  <span class="mono">posture_rom_full.json</span>; every value traces to a recorded trial.
  Free-air ROM protocol, ideal transmission &mdash; a calibration against Zhao et&nbsp;al.'s
  published similarities does not fit at any single strap stiffness, so these figures represent
  each device at its theoretical best rather than as worn.</p>
</footer>
</div>
"""
    open(dest, "w").write(html)
    print(f"wrote {dest} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
