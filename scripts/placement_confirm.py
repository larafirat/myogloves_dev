import sys, json
sys.path.insert(0,'myogloves_dev/scripts')
import hold_benchmark as hb
NAMES=["healthy_box","glove_dev_box",
       "portOP_D1_underactuated_distal_calibrated","portOP_D2_synergy_cross_finger_calibrated",
       "portOP_D3_uniform_single_dof_calibrated","portOP_D4_v2_hybrid_per_finger_calibrated"]
out={}
for n in NAMES:
    ad=hb.glove_dev_adapters()[n]; ad.pos_override=(0.09,0.10,0.22)
    res=[hb.run_trial(ad,seed=1000+i,t_max=hb.T_MAX_DEFAULT) for i in range(15)]
    s=hb.summarise(res); out[n]=s
    print(hb.fmt_summary(ad.name,s),flush=True)
    json.dump(out,open('myogloves_dev/placement_confirm_fixed.json','w'),indent=2)
