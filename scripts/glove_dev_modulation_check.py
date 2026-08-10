"""Does giving glove_dev the same contact modulation every K-matrix device has
change its performance? Also re-tests the original script's HOLD_FRACTION=1.0
conclusion, which predates the friction, placement and drop-test corrections.
"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
SET=[(None,"open loop (shipped)"),(0.5,"ease-off 0.50"),(0.3,"ease-off 0.30"),(0.15,"ease-off 0.15")]
for obj in ("box","can","tuna"):
    print(f"\n=== {obj.upper()}   glove_dev   survival%(grasp%) | hold_med   15 trials", flush=True)
    for hf,label in SET:
        hb.GLOVE_DEV_HOLD_FRACTION=hf
        res=[hb.run_trial(hb.glove_dev_adapters()[f"glove_dev_{obj}"],1000+i,5.0) for i in range(15)]
        s=hb.summarise(res)
        print(f"   {label:22} {s['survival_rate']*100:5.0f}% ({s['grasp_rate']*100:3.0f}) "
              f"{s['hold_median']:5.2f}s   [NG={s['no_grasp']} SL={s['slipped']} SV={s['survived']}]", flush=True)
