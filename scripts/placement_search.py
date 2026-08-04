"""Per-device placement search. The box position (0.07, 0.10, 0.22) was found
for D4 and reused for everyone, so any device that prefers a different pose has
been scored at someone else's optimum."""
import sys, json, itertools
sys.path.insert(0,'myogloves_dev/scripts')
import hold_benchmark as hb

XS=[0.05,0.07,0.09]; YS=[0.08,0.10,0.12]; ZS=[0.20,0.22,0.24]
SCREEN_TRIALS=3
DEVICES=["glove_dev_box",
         "portOP_D1_underactuated_distal_calibrated",
         "portOP_D2_synergy_cross_finger_calibrated",
         "portOP_D3_uniform_single_dof_calibrated",
         "portOP_D4_v2_hybrid_per_finger_calibrated"]

pool=hb.glove_dev_adapters()
out={}
for name in DEVICES:
    ad=pool[name]; ranked=[]
    for z,x,y in itertools.product(ZS,XS,YS):
        ad.pos_override=(x,y,z)
        res=[hb.run_trial(ad,seed=2000+i,t_max=hb.T_MAX_DEFAULT) for i in range(SCREEN_TRIALS)]
        s=hb.summarise(res)
        ranked.append((s["hold_median"],s["grasp_rate"],x,y,z))
        print(f"{ad.name:28s} ({x},{y},{z}) grasp={s['grasp_rate']*100:5.1f}% hold={s['hold_median']:5.2f}s",flush=True)
    ranked.sort(reverse=True)
    out[name]=[{"hold":h,"grasp":g,"pos":[x,y,z]} for h,g,x,y,z in ranked]
    print(f"  >>> {ad.name} best: {ranked[0]}",flush=True)
    json.dump(out,open('myogloves_dev/placement_screen.json','w'),indent=2)
