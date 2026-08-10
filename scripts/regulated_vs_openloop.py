"""Open-loop vs force-regulated, the SAME rule applied to every device.

GRIP_SAFETY_MARGIN=5 is a fitted constant, not a derived one: an attempt to
take it from the healthy hand's own grip ratio failed, because the healthy hand
holds at ~150x its slip threshold (141 N on a 0.95 N box). Reported as a
declared shared setting rather than dressed up as principled.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.GRIP_SAFETY_MARGIN=5.0
CON=[("healthy","healthy_%s"),("Tyrone","glove_dev_%s"),
     ("D1","portOP_D1_underactuated_distal_calibrated%s"),
     ("D2","portOP_D2_synergy_cross_finger_calibrated%s"),
     ("D3","splint_D3%s"),
     ("D4","portOP_D4_v2_hybrid_per_finger_calibrated%s")]
def key(pat,obj):
    if "%s" in pat and ("healthy" in pat or "glove_dev" in pat): return pat % obj
    return pat % ("" if obj=="box" else "_"+obj)
for obj in ("box","can","tuna"):
    print(f"\n=== {obj.upper()}   survival%(grasp%) | hold   15 trials", flush=True)
    print(f"{'':9}{'open loop':>26}{'force-regulated':>26}", flush=True)
    for label,pat in CON:
        row=""
        for on in (False,True):
            hb.FORCE_EASE_OFF=on
            res=[hb.run_trial(hb.glove_dev_adapters()[key(pat,obj)],1000+i,5.0) for i in range(15)]
            s=hb.summarise(res)
            row+=f"{s['survival_rate']*100:11.0f}%({s['grasp_rate']*100:3.0f}) {s['hold_median']:5.2f}s"
        print(f"{label:9}{row}", flush=True)
