"""Does the measurement window change the answer?

myoMPL (Tan et al., MyoAssist 0.1) runs episodes of 10 s and requires 1 s of
contact -- our gate already matches the 1 s. Ours cuts at 5 s. Tested at a mass
where the devices actually separate.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
CON=[("healthy","healthy_box"),("Tyrone","glove_dev_box"),
     ("D1 +OP","portOP_D1_underactuated_distal_calibrated"),
     ("D2 +OP","portOP_D2_synergy_cross_finger_calibrated"),
     ("D4 +OP","portOP_D4_v2_hybrid_per_finger_calibrated")]
for mass in (0.7, 1.2):
    print(f"\n=== {mass} kg   survival%   12 trials", flush=True)
    print(f"{'':10}{'T=5s':>10}{'T=10s':>10}{'T=20s':>10}", flush=True)
    for lbl,key in CON:
        row=""
        for T in (5.0,10.0,20.0):
            res=[hb.run_trial(hb.glove_dev_adapters()[key],1000+i,T,mass_override=mass) for i in range(12)]
            s=hb.summarise(res)
            row+=f"{s['survival_rate']*100:9.0f}%"
        print(f"{lbl:10}{row}", flush=True)
