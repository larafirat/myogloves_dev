"""Does correcting EXO_FLEX's lengthrange raise the glove's load capacity?"""
import sys, os
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
M=[0.1,0.35,0.7,1.2,2.0]
print("glove_dev on the box, survival%(grasp%), 12 trials per cell\n")
print(f"{'':26}"+"".join(f"{str(m)+' kg':>14}" for m in M))
for lr,label in ((None,"as declared"),((0.2217,0.2861),"lengthrange corrected")):
    hb.EXO_FLEX_LENGTHRANGE=lr
    row=""
    for mass in M:
        ad=hb.glove_dev_adapters()['glove_dev_box']
        s=hb.summarise([hb.run_trial(ad,1000+i,10.0,mass_override=mass) for i in range(12)])
        row+=f"{s['survival_rate']*100:8.0f}%({s['grasp_rate']*100:3.0f})"
    print(f"{label:26}{row}", flush=True)
# healthy for reference, unaffected by the override
hb.EXO_FLEX_LENGTHRANGE=None
row=""
for mass in M:
    s=hb.summarise([hb.run_trial(hb.glove_dev_adapters()['healthy_box'],1000+i,10.0,mass_override=mass) for i in range(12)])
    row+=f"{s['survival_rate']*100:8.0f}%({s['grasp_rate']*100:3.0f})"
print(f"{'healthy hand (reference)':26}{row}", flush=True)
