"""Does giving D1-D4 live moment arms change the answer?"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
from exo_devices import JOINT_NAMES

# how far the arms actually drift during a closing motion
hb.LIVE_MOMENT_ARMS=True
ad=hb.glove_dev_adapters()['portOP_D4_v2_hybrid_per_finger_calibrated']
m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
hb.settle(m,d,ad.oid,ad.settle_cap)
ramp=max(1,int(ad.closing_seconds/m.opt.timestep)); ext={}
for k in range(int(4.0/m.opt.timestep)):
    ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    s=ad.live_arms.scale(d,len(JOINT_NAMES))
    for r,j in enumerate(JOINT_NAMES):
        if s[r]!=1.0 or j in ext:
            lo,hi=ext.get(j,(9,-9)); ext[j]=(min(lo,s[r]),max(hi,s[r]))
print("moment-arm multiplier range during a D4 grasp (1.00 = the frozen value):")
for j,(lo,hi) in ext.items():
    if abs(hi-1)>0.05 or abs(lo-1)>0.05:
        print(f"   {j:15s} {lo:5.2f} .. {hi:5.2f}")

print("\nsurvival%(grasp%) | hold_med       10 trials")
print(f"{'':34}{'frozen arms':>22}{'live arms':>22}")
for label,key in (("Tyrone (unaffected)","glove_dev_box"),
                  ("D1 box","portOP_D1_underactuated_distal_calibrated"),
                  ("D2 box","portOP_D2_synergy_cross_finger_calibrated"),
                  ("D4 box","portOP_D4_v2_hybrid_per_finger_calibrated"),
                  ("D1 can","portOP_D1_underactuated_distal_calibrated_can"),
                  ("D4 can","portOP_D4_v2_hybrid_per_finger_calibrated_can")):
    row=""
    for live in (False,True):
        hb.LIVE_MOMENT_ARMS=live
        res=[hb.run_trial(hb.glove_dev_adapters()[key],1000+i,5.0) for i in range(10)]
        s=hb.summarise(res)
        row+=f"{s['survival_rate']*100:7.0f}%({s['grasp_rate']*100:3.0f}) {s['hold_median']:5.2f}s"
    print(f"{label:34}{row}", flush=True)
