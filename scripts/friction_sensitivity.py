"""Does the DEVICE RANKING depend on glove friction?

If the ordering is stable across the plausible glove range, per-device friction
values are not worth inventing. If it flips, friction must be held constant and
declared, because otherwise a device wins on fabric rather than mechanism.

Object stays at the MyoAssist reference [1.0, 0.005, 0.0001] throughout, so the
benchmark object is never altered. Only the glove side moves. Note MuJoCo
combines friction as element-wise MAX at equal geom priority, so a glove value
below the object's 1.0 has no effect -- 1.0 IS the bare-hand floor here.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

GLOVE = [1.0, 1.5, 2.0, 2.5]
CONDS = [("healthy","healthy_box","healthy_can"),
         ("Tyrone","glove_dev_box","glove_dev_can"),
         ("D1","portOP_D1_underactuated_distal_calibrated","portOP_D1_underactuated_distal_calibrated_can"),
         ("D2","portOP_D2_synergy_cross_finger_calibrated","portOP_D2_synergy_cross_finger_calibrated_can"),
         ("D3","splint_D3","splint_D3_can"),
         ("D4","portOP_D4_v2_hybrid_per_finger_calibrated","portOP_D4_v2_hybrid_per_finger_calibrated_can")]

for oi, obj in ((1,"BOX"),(2,"CAN")):
    print(f"\n=== {obj}   survival% (grasp%)   object friction fixed at MyoAssist 1.0")
    print(f"{'device':<9}" + "".join(f"{'glove '+str(g):>16}" for g in GLOVE))
    for label, *keys in CONDS:
        row=""
        for g in GLOVE:
            hb.GLOVE_FRICTION = [g, 0.005, 0.0001]
            res=[hb.run_trial(hb.glove_dev_adapters()[keys[oi-1]], 1000+i, 5.0) for i in range(10)]
            s=hb.summarise(res)
            row += f"{s['survival_rate']*100:9.0f}% ({s['grasp_rate']*100:3.0f})"
        print(f"{label:<9}{row}", flush=True)
