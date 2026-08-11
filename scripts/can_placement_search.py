import sys, os, itertools, numpy as np
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.STABILISE_WRIST=True
def q(axis,ang):
    a=np.array(axis,float); a/=np.linalg.norm(a); s=np.sin(ang/2)
    return [np.cos(ang/2), a[0]*s, a[1]*s, a[2]*s]
QUAT=q([0,1,0],np.pi/2)   # can axis along world x
print("can lying, axis along x -- refining around (0.095,0.020,0.120)")
best=None
for z,x,y in itertools.product([0.105,0.120,0.135],[0.085,0.095,0.105],[0.005,0.020,0.035]):
    ad=hb.glove_dev_adapters()['healthy_can']
    ad.pos_override=(x,y,z); ad.quat_override=QUAT
    res=[hb.run_trial(ad,2000+i,5.0) for i in range(6)]
    s=hb.summarise(res)
    if s['grasp_rate']>0:
        fl=f" SETUPFAIL={s['setup_fail']}" if s['setup_fail'] else ""
        print(f"   ({x},{y},{z}) grasp={s['grasp_rate']*100:5.0f}% surv={s['survival_rate']*100:5.0f}% "
              f"hold={s['hold_median']:5.2f}s{fl}", flush=True)
    k=(s['survival_rate'],s['grasp_rate'],s['hold_median'])
    if best is None or k>best[0]: best=(k,(x,y,z))
print(f"   >>> best {best[1]} surv={best[0][0]*100:.0f}% grasp={best[0][1]*100:.0f}% hold={best[0][2]:.2f}s")
