"""What did disabling the 25x hand damping cost Tyrone's glove, and does it
still cripple the K-matrix devices at the current friction and placements?

The claim recorded when it was switched off was "glove_dev is unaffected by
removing it". That was measured at legacy friction and the old placements.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

DAMP=[1.0, 5.0, 10.0, 25.0]
CONDS=[("healthy","healthy_box","healthy_can"),
       ("Tyrone","glove_dev_box","glove_dev_can"),
       ("D1","portOP_D1_underactuated_distal_calibrated","portOP_D1_underactuated_distal_calibrated_can"),
       ("D4","portOP_D4_v2_hybrid_per_finger_calibrated","portOP_D4_v2_hybrid_per_finger_calibrated_can")]
for oi,obj in ((1,"BOX"),(2,"CAN")):
    print(f"\n=== {obj}   survival%(grasp%) | hold_med   at reference friction")
    print(f"{'':8}"+"".join(f"{'damp x'+str(int(d)):>22}" for d in DAMP))
    for label,*keys in CONDS:
        row=""
        for dmp in DAMP:
            hb.HAND_DAMPING_BOOST=dmp
            res=[hb.run_trial(hb.glove_dev_adapters()[keys[oi-1]],1000+i,5.0) for i in range(10)]
            s=hb.summarise(res)
            row+=f"{s['survival_rate']*100:7.0f}%({s['grasp_rate']*100:3.0f}) {s['hold_median']:5.2f}s"
        print(f"{label:8}{row}", flush=True)
