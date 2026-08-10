"""Margin 5 was fitted on ONE condition (glove_dev/can). Does it generalise?

If a single shared constant works across three objects and five devices, it is
capturing something real about the contact geometry. If it only helps the
condition it was fitted on, it is overfitting and should be rejected.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.GRIP_SAFETY_MARGIN=5.0
CON=[("healthy","healthy_%s"),("Tyrone","glove_dev_%s"),
     ("D1","portOP_D1_underactuated_distal_calibrated%s"),
     ("D2","portOP_D2_synergy_cross_finger_calibrated%s"),
     ("D4","portOP_D4_v2_hybrid_per_finger_calibrated%s")]
for obj in ("box","can","tuna"):
    print(f"\n=== {obj.upper()}   survival%(grasp%) | hold_med   12 trials", flush=True)
    print(f"{'':10}{'open loop':>24}{'force ease-off':>24}", flush=True)
    for label,pat in CON:
        key = pat % (obj if "healthy" in pat or "glove_dev" in pat else ("" if obj=="box" else "_"+obj))
        row=""
        for on in (False,True):
            hb.FORCE_EASE_OFF=on
            try:
                res=[hb.run_trial(hb.glove_dev_adapters()[key],1000+i,5.0) for i in range(12)]
            except KeyError:
                row+=f"{'n/a':>24}"; continue
            s=hb.summarise(res)
            row+=f"{s['survival_rate']*100:9.0f}%({s['grasp_rate']*100:3.0f}) {s['hold_median']:5.2f}s"
        print(f"{label:10}{row}", flush=True)
