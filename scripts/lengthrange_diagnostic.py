"""DIAGNOSTIC ONLY. What is the EXO_FLEX lengthrange mismatch costing?

Builds a THROWAWAY copy of the model with exo_flex_tendon's lengthrange set to
the range the tendon can actually reach (0.2217-0.2861 m, measured by sampling
the joints it crosses). The shipped file is never modified and the copy is
deleted. This only measures the size of the effect so it can be decided on.
"""
import sys, os, numpy as np
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

BASE='myogloves_dev/models/myohand_glove_dev%s.xml'
TMP ='myogloves_dev/models/_tmp_lr%s.xml'
OBJ={'':('009_gelatin_box',0.036,0.014,0.097,-0.044), '_can':('005_tomato_soup_can',0.033,0.05,0.349,0.0)}
made=[]
try:
    for suf in ('','_can'):
        s=open(BASE%suf).read()
        old='<muscle name="EXO_FLEX" force="204.8" class="muscle" tendon="exo_flex_tendon" lengthrange="0.3400 0.7800"/>'
        new='<muscle name="EXO_FLEX" force="204.8" class="muscle" tendon="exo_flex_tendon" lengthrange="0.2217 0.2861"/>'
        assert s.count(old)==1, s.count(old)
        s=s.replace(old,new)
        open(TMP%suf,'w').write(s); made.append(TMP%suf)
    for suf,label in (('','BOX'),('_can','CAN')):
        name,r,hh,mass,off = OBJ[suf]
        for path,tag in ((BASE%suf,'as shipped  '),(TMP%suf,'lengthrange fixed')):
            ad=hb.GloveDevAdapter(path,name,r,hh,mass,f"{tag}",bottom_offset=off)
            res=[hb.run_trial(ad,1000+i,5.0) for i in range(10)]
            st=hb.summarise(res)
            # peak force actually delivered
            import mujoco
            m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
            hb.settle(m,d,ad.oid,ad.settle_cap)
            ramp=max(1,int(ad.closing_seconds/m.opt.timestep)); pk=0.0
            for k in range(int(3.0/m.opt.timestep)):
                ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
                pk=max(pk,abs(float(d.actuator_force[m.actuator('EXO_FLEX').id])))
            print(f"{label:4} {tag:18} grasp={st['grasp_rate']*100:5.1f}%  "
                  f"surv={st['survival_rate']*100:5.1f}%  hold={st['hold_median']:5.2f}s   "
                  f"peak EXO_FLEX = {pk:6.1f} N ({pk/204.8:5.1%} of rating)", flush=True)
finally:
    for p in made:
        if os.path.exists(p): os.remove(p)
    print("\n(throwaway copies deleted; myohand_glove_dev*.xml untouched)")
