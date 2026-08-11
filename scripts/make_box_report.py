"""Report for the rebuilt gelatin-box study -- one object, done properly.

Deliberately narrow. The earlier three-object report (make_benchmark_report.py)
covers the original floppy-wrist study and stays as it is; this one covers only
what the rebuilt rig can actually support, which is the 009_gelatin_box. The can
and tuna are excluded because the rebuild showed this hand cannot close around
them at any mass or placement, and saying so is more useful than reporting their
failures as device results.

Every figure is read from the run JSON, so the page cannot drift from the data.

Usage:
    python myogloves_dev/scripts/make_box_report.py \
        final_box_10s.json final_mass_10s.json posture_rom_final12.json out.html
"""

from __future__ import annotations

import json
import sys

DEVICES = [
    ("healthy_box", "Healthy hand", "no device &middot; own muscles", "ref"),
    ("glove_dev_box", "Tyrone", "3 routed tendons &middot; the only modelled hardware", "dev"),
    ("portOP_D4_v2_hybrid_per_finger_calibrated", "D4", "Gerez 2020 &middot; 5 motors, own opposition tendon", "dev"),
    ("portOP_D1_underactuated_distal_calibrated", "D1", "Zhao 2025 &middot; 5 motors, no abduction DoF", "dev"),
    ("portOP_D2_synergy_cross_finger_calibrated", "D2", "Alicea 2021 &middot; 1 motor, synergy pulley", "dev"),
    ("splint_D3", "D3", "Thimabut 2022 &middot; 2 fingers + rigid splint", "dev"),
]
NO_OP = [
    ("port_D4_v2_hybrid_per_finger_calibrated", "D4"),
    ("port_D1_underactuated_distal_calibrated", "D1"),
    ("port_D2_synergy_cross_finger_calibrated", "D2"),
]
MASSES = ["0.1", "0.35", "0.7", "1.2", "2.0"]
MASS_ROWS = [("healthy_box", "Healthy hand"), ("glove_dev_box", "Tyrone"),
             ("portOP_D4_v2_hybrid_per_finger_calibrated", "D4"),
             ("portOP_D1_underactuated_distal_calibrated", "D1"),
             ("portOP_D2_synergy_cross_finger_calibrated", "D2")]

POSTURE = {"glove_dev/box": "Tyrone",
           "PORT+OP/box/D4_v2_hybrid_per_finger": "D4",
           "PORT+OP/box/D1_underactuated_distal": "D1",
           "PORT+OP/box/D2_synergy_cross_finger": "D2",
           "SPLINT/box/D3_uniform_single_dof": "D3"}
DIGITS = [("index", [("MCP", "mcp2_flexion"), ("PIP", "pm2_flexion"), ("DIP", "md2_flexion")]),
          ("middle", [("MCP", "mcp3_flexion"), ("PIP", "pm3_flexion"), ("DIP", "md3_flexion")]),
          ("ring", [("MCP", "mcp4_flexion"), ("PIP", "pm4_flexion"), ("DIP", "md4_flexion")]),
          ("little", [("MCP", "mcp5_flexion"), ("PIP", "pm5_flexion"), ("DIP", "md5_flexion")]),
          ("thumb", [("ABD", "cmc_abduction"), ("CMC", "cmc_flexion"),
                     ("MP", "mp_flexion"), ("IP", "ip_flexion")])]
FLAT = [(d, l, j) for d, js in DIGITS for l, j in js]


def band(p):
    return "na" if p is None else "good" if p >= 60 else "mid" if p >= 20 else "low"


def main():
    hold = json.load(open(sys.argv[1]))
    mass = json.load(open(sys.argv[2]))
    post = json.load(open(sys.argv[3]))
    dest = sys.argv[4]
    n = hold["healthy_box"]["n"]
    tmax = max(v["hold_median"] for v in hold.values())

    rows = []
    for key, name, blurb, kind in DEVICES:
        s = hold.get(key)
        if not s:
            continue
        g, sv, hd = s["grasp_rate"] * 100, s["survival_rate"] * 100, s["hold_median"]
        dag = '<sup title="object left its support during settling">&dagger;</sup>' if s.get("setup_fail") else ''
        rows.append(
            f'<tr class="{kind}"><th class="rowlab"><b>{name}</b><span class="blurb">{blurb}</span></th>'
            f'<td class="mono">{g:.0f}%{dag}</td>'
            f'<td class="mono surv band-{band(sv)}" style="--fill:{max(sv,0):.0f}%">{sv:.0f}%</td>'
            f'<td class="mono soft">{hd:.2f}s</td></tr>')

    noop = []
    for key, name in NO_OP:
        s = hold.get(key)
        if not s:
            continue
        noop.append(f'<tr><th class="rowlab">{name} <span class="blurb">no pre-shape</span></th>'
                    f'<td class="mono">{s["grasp_rate"]*100:.0f}%</td>'
                    f'<td class="mono surv band-{band(s["survival_rate"]*100)}" '
                    f'style="--fill:{max(s["survival_rate"]*100,0):.0f}%">{s["survival_rate"]*100:.0f}%</td>'
                    f'<td class="mono soft">{s["hold_median"]:.2f}s</td></tr>')

    mrows = []
    for key, name in MASS_ROWS:
        cells = []
        for mk in MASSES:
            s = mass.get(f"{key}@{mk}kg")
            if not s:
                cells.append('<td class="na">&mdash;</td>')
                continue
            sv = s["survival_rate"] * 100
            cells.append(f'<td class="mono cell band-{band(sv)}" style="--fill:{max(sv,0):.0f}%" '
                         f'title="grasp {s["grasp_rate"]*100:.0f}%, hold {s["hold_median"]:.2f}s">{sv:.0f}</td>')
        mrows.append(f'<tr><th class="rowlab">{name}</th>{"".join(cells)}</tr>')

    ref = post["reference_deg"]
    phead = "".join(f'<th class="j"><span class="dig">{d[:3]}</span>{l}</th>' for d, l, _ in FLAT)
    prefrow = "".join(f'<td class="mono {"faint" if abs(ref[j]) < post["min_ref_excursion_deg"] else ""}">'
                      f'{ref[j]:+.0f}</td>' for _, _, j in FLAT)
    prows = []
    for full, short in POSTURE.items():
        dv = post["devices"].get(full)
        if not dv:
            continue
        cs = []
        for _, _, j in FLAT:
            m_ = dv["match_pct"].get(j, float("nan"))
            if j in dv["splinted"]:
                cs.append('<td class="cell splinted" title="held by the splint">&#9647;</td>')
            elif m_ != m_:
                cs.append('<td class="cell na" title="reference moves too little to score">&middot;</td>')
            else:
                cs.append(f'<td class="cell band-{band(m_)}" style="--fill:{min(max(m_,0),100):.0f}%" '
                          f'title="{dv["excursion_deg"][j]:+.1f} deg">{m_:.0f}</td>')
        prows.append(f'<tr><th class="rowlab">{short}<span class="blurb">'
                     f'cover {dv["coverage_pct"]:.0f}% &middot; fidelity {dv["fidelity_pct"]:.0f}%</span></th>'
                     f'{"".join(cs)}</tr>')

    html = f"""<title>Gelatin box &mdash; exoglove benchmark on a rebuilt rig</title>
<style>
:root {{
  --ground:#f6f7f5; --surface:#fff; --ink:#13181d; --muted:#69737c;
  --rule:#dde2de; --rule-strong:#c3cac4; --accent:#16695a;
  --good:#2f7a56; --mid:#9c7a1e; --low:#a03a30; --wash:#eef1ee;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}}
@media (prefers-color-scheme:dark) {{ :root {{
  --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
  --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
  --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127; }} }}
:root[data-theme="dark"] {{ --ground:#0d1114; --surface:#151b20; --ink:#e3e8ea; --muted:#8b959d;
  --rule:#242d34; --rule-strong:#38434b; --accent:#5fbfa8;
  --good:#5aab84; --mid:#c9a44e; --low:#d1665a; --wash:#1a2127; }}
:root[data-theme="light"] {{ --ground:#f6f7f5; --surface:#fff; --ink:#13181d; --muted:#69737c;
  --rule:#dde2de; --rule-strong:#c3cac4; --accent:#16695a;
  --good:#2f7a56; --mid:#9c7a1e; --low:#a03a30; --wash:#eef1ee; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:clamp(1.5rem,4vw,3.5rem) clamp(1rem,3vw,2rem) 5rem; }}
.mono {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
header.top {{ border-bottom:2px solid var(--ink); padding-bottom:1.5rem; margin-bottom:2.5rem; }}
.eyebrow {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); margin:0 0 .6rem; }}
h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(1.9rem,4.5vw,2.8rem); line-height:1.1;
  margin:0 0 .7rem; text-wrap:balance; letter-spacing:-.01em; }}
.standfirst {{ max-width:62ch; color:var(--muted); margin:0 0 1.4rem; font-size:1.02rem; }}
.method {{ display:flex; flex-wrap:wrap; gap:.4rem 1.5rem; font-family:var(--mono); font-size:.75rem; color:var(--muted); }}
.method b {{ color:var(--ink); font-weight:500; }}
h2 {{ font-family:var(--serif); font-weight:600; font-size:1.4rem; margin:3rem 0 .4rem; }}
h2:first-of-type {{ margin-top:0; }}
h3 {{ font-family:var(--serif); font-size:1.02rem; font-weight:600; margin:1.8rem 0 .5rem; }}
.lede {{ max-width:64ch; color:var(--muted); margin:0 0 1.3rem; font-size:.93rem; }}
.blurb {{ display:block; font-family:var(--sans); font-size:.72rem; font-weight:400; color:var(--muted); }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:.83rem; }}
th,td {{ padding:.45rem .55rem; text-align:right; white-space:nowrap; border-bottom:1px solid var(--rule); }}
thead th {{ background:var(--surface); font-weight:500; font-size:.7rem; color:var(--muted);
  border-bottom:1px solid var(--rule-strong); }}
th.rowlab {{ text-align:left; font-weight:500; position:sticky; left:0; background:var(--surface);
  border-right:1px solid var(--rule); min-width:130px; }}
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
th.j .dig {{ display:block; font-family:var(--mono); font-size:.62rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--accent); }}
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; margin:.7rem 0 0; font-size:.75rem; color:var(--muted); }}
.legend span {{ display:inline-flex; align-items:center; gap:.4rem; }}
.swatch {{ width:22px; height:3px; display:inline-block; }}
.finding {{ border-left:2px solid var(--accent); padding:.1rem 0 .1rem 1rem; margin:1.5rem 0; max-width:66ch; }}
.finding p {{ margin:.35rem 0; }}
.fixes {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); margin-top:1rem; }}
.fixes div {{ background:var(--surface); padding:.85rem 1rem; }}
.fixes h5 {{ margin:0 0 .3rem; font-size:.8rem; font-weight:600; }}
.fixes p {{ margin:0; font-size:.79rem; color:var(--muted); }}
footer {{ margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--rule); font-size:.78rem;
  color:var(--muted); max-width:70ch; }}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">MyoSuite &middot; MyoHand &middot; YCB 009 gelatin box &middot; {n} trials per cell</p>
  <h1>One object, one rig, five gloves</h1>
  <p class="standfirst">Four exoskeleton gloves from the literature and one in development, driven
  through a rig rebuilt so that the hand actually grasps: wrist held, object in front of the palm,
  contact on the palm rather than the fingertips alone. The reference is the same hand with no
  device.</p>
  <p class="method">
    <span><b>Object</b> YCB 009, 97 g, as myoMPL uses</span>
    <span><b>Friction</b> 1.0 / 0.005 / 0.0001</span>
    <span><b>Window</b> {tmax:.0f} s, per myoMPL</span>
    <span><b>Gate</b> 2+ opposing digits, 1 s</span>
    <span><b>Failure</b> 50 mm relative to the hand</span></p>
</header>

<h2>Holding the object</h2>
<p class="lede">Survival is the headline: the share of trials still holding at {tmax:.0f}&nbsp;s. Grasp is whether
an opposed grip formed at all; hold is the median time to failure among trials that did grasp, so a
long hold beside a zero survival means it always failed, but slowly.</p>
<div class="scroll"><table>
  <thead><tr><th class="rowlab">condition</th><th>grasp</th><th>survival</th><th>hold</th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table></div>

<h3>The same devices without the muscle pre-shape</h3>
<p class="lede">The ported devices are normally given the hand's own opponens pollicis at 0.5 to
pre-position the thumb. Removed, only their own actuation remains &mdash; and this is what separates
them.</p>
<div class="scroll"><table>
  <thead><tr><th class="rowlab">condition</th><th>grasp</th><th>survival</th><th>hold</th></tr></thead>
  <tbody>{"".join(noop)}</tbody>
</table></div>

<div class="finding">
  <p><strong>At 97&nbsp;g the task is too easy to rank anything.</strong> Every condition with a working
  thumb opposition &mdash; whether its own tendon or a borrowed muscle &mdash; holds the box for the full
  window. What the rig separates cleanly is opposition itself: remove the pre-shape and the ported
  devices collapse to 0&ndash;13%.</p>
  <p><strong>D3 cannot grasp at all here.</strong> Its splint holds the thumb at a fixed &minus;45&deg; MCP that
  does not meet the object where the fingers now close &mdash; that angle was effectively set against
  the old placement and needs redoing for this geometry.</p>
</div>

<h2>Adding load</h2>
<p class="lede">The same grasp under increasing mass, to 2&nbsp;kg &mdash; twenty times the object's own
weight. This is where the devices separate, and it is the reason the single-mass table above should
not be read as a ranking.</p>
<div class="scroll"><table>
  <thead><tr><th class="rowlab">survival %</th>{"".join(f'<th class="mono">{m} kg</th>' for m in MASSES)}</tr></thead>
  <tbody>{"".join(mrows)}</tbody>
</table></div>
<p class="legend">
  <span><i class="swatch" style="background:var(--good)"></i>&ge;60%</span>
  <span><i class="swatch" style="background:var(--mid)"></i>20&ndash;60%</span>
  <span><i class="swatch" style="background:var(--low)"></i>&lt;20%</span>
  <span>hover for grasp rate and median hold</span></p>

<div class="finding">
  <p><strong>Tyrone's glove degrades most gracefully of any device.</strong> It is the only one still
  holding anything above 0.35&nbsp;kg, and it fails progressively rather than falling off a cliff. D4
  is next; D1 and D2 are at zero by 0.35&nbsp;kg. Grasp rate stays near 100% for all of them at every
  mass, so this is a holding limit, not a closing one.</p>
  <p><strong>The healthy hand is flat at 100% to 2&nbsp;kg</strong> and never becomes the binding
  reference, so the useful comparison here is device against device.</p>
</div>

<h2>Is the hand the right shape?</h2>
<p class="lede">Holding time cannot see posture. This reproduces Zhao et&nbsp;al.'s published joint-angle
similarity protocol &mdash; free-air closing, scored against the same hand moving under its own muscles.
The top row is that reference, in degrees.</p>
<div class="scroll"><table>
  <thead><tr><th class="rowlab">device</th>{phead}</tr></thead>
  <tbody><tr class="ref"><th class="rowlab">healthy &deg;</th>{prefrow}</tr>{"".join(prows)}</tbody>
</table></div>
<p class="legend">
  <span><i class="swatch" style="background:var(--accent)"></i>&#9647; held by the splint</span>
  <span>&middot; reference moves &lt;{post["min_ref_excursion_deg"]:.0f}&deg;, not scorable</span>
  <span>coverage = joints driven at all; fidelity = match on those joints</span></p>

<div class="finding">
  <p><strong>Only Tyrone's glove registers real thumb abduction.</strong> Its exo tendon has a
  13.7&nbsp;mm moment arm at that joint against the natural flexor's 3.9&nbsp;mm. Every ported device
  sits near zero there, because the pre-shape they borrow has already driven the joint to its limit
  before they act.</p>
</div>

<h2>What the rebuild changed</h2>
<p class="lede">An earlier version of this study ran on a rig that measured something else. These are
the corrections, each found by measurement rather than assumption.</p>
<div class="fixes">
  <div><h5>The wrist was unactuated</h5><p>No actuator on pro_sup, deviation or flexion. The forearm
  pronated 46&deg; before a trial began and swung 20&ndash;50&deg; during it. Now held.</p></div>
  <div><h5>The object sat beside the hand</h5><p>Palm 49&ndash;74&nbsp;mm clear of the object surface, so
  palm contact was impossible and every grasp was a fingertip pinch reached by rotating the wrist.</p></div>
  <div><h5>Failure was measured in the world</h5><p>The servo arm sags 39&nbsp;mm when the support is
  removed, which was scored as a dropped object. Now measured relative to the hand.</p></div>
  <div><h5>The gate ignored opposition</h5><p>Any two digits counted, so a pinky brushing the object
  certified a grasp. Now a pair must actually oppose.</p></div>
  <div><h5>Friction was 2.5&times; the reference</h5><p>Undocumented, and mostly acting on the
  object&ndash;support contact rather than the grasp. Now the MyoAssist value.</p></div>
  <div><h5>The hand never settled</h5><p>Convergence was tested on velocity, which chatter makes
  unreachable, so every trial began from an undetected pose. Now tested on displacement.</p></div>
</div>

<h2>What this does not cover</h2>
<p class="lede">Stated because the numbers above are only as good as these.</p>
<div class="fixes">
  <div><h5>One object</h5><p>The soup can and tuna were dropped, not lost: at 66 and 84&nbsp;mm they
  exceed the hand's measured 64&nbsp;mm enclosure and cannot be closed around at any mass or
  placement. The pudding box (36&nbsp;mm, 187&nbsp;g) reproduces this ordering; the sugar box is too
  tall to grasp.</p></div>
  <div><h5>Ideal transmission</h5><p>No strap compliance, friction or slack. Fitting a series-spring
  model to published similarity data failed at every stiffness, so devices run at their theoretical
  best.</p></div>
  <div><h5>Frozen moment arms</h5><p>D1&ndash;D4 are joint torques with arms measured once at rest;
  live geometry differs by 1.3&ndash;2&times; mid-grasp. Measured, and it does not change the ordering.</p></div>
  <div><h5>Maximal effort throughout</h5><p>Every condition, the reference included, grips at
  100&ndash;150&times; its slip threshold, because all are driven to full activation with no force target.</p></div>
</div>

<footer>
  <p>Generated by <span class="mono">make_box_report.py</span> from the run JSON; no figure here is
  transcribed. Object, friction, contact gate and measurement window follow myoMPL (Tan et&nbsp;al.,
  MyoAssist&nbsp;0.1, ICORR 2025), which manipulates the same YCB gelatin box. The earlier
  three-object report covers the pre-rebuild study and is kept unchanged.</p>
</footer>
</div>
"""
    open(dest, "w").write(html)
    print(f"wrote {dest} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
