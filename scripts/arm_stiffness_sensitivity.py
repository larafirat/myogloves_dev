"""How much does the compliant servo arm cost each device?

Tyrone's original rig has NO arm -- myohand_tabletop_dev.xml mounts the hand
directly. myoMPL fixes its torso. Ours interposes a 6-DOF position servo that
sags 39 mm at release and can absorb grasp energy. The arm is a FIXTURE, so
unlike damping or friction there is no fairness objection to making it rigid.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

CONDS=[("healthy","healthy_box","healthy_can"),
       ("Tyrone","glove_dev_box","glove_dev_can"),
       ("D1","portOP_D1_underactuated_distal_calibrated","portOP_D1_underactuated_distal_calibrated_can"),
       ("D4","portOP_D4_v2_hybrid_per_finger_calibrated","portOP_D4_v2_hybrid_per_finger_calibrated_can")]
STIFF=[8.0, 40.0, 200.0]
for oi,obj in ((1,"BOX"),(2,"CAN")):
    print(f"\n=== {obj}   survival%(grasp%) | hold_med", flush=True)
    print(f"{'':8}"+"".join(f"{'arm kp x'+str(int(s)):>22}" for s in STIFF), flush=True)
    for label,*keys in CONDS:
        row=""
        for st in STIFF:
            hb.ARM_STIFF_BOOST=st
            res=[hb.run_trial(hb.glove_dev_adapters()[keys[oi-1]],1000+i,5.0) for i in range(10)]
            s=hb.summarise(res)
            row+=f"{s['survival_rate']*100:7.0f}%({s['grasp_rate']*100:3.0f}) {s['hold_median']:5.2f}s"
        print(f"{label:8}{row}", flush=True)
