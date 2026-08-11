"""Cap DIP flexion directly.

A passive spring cannot do this: the drives differ by orders of magnitude, so
one stiffness gives the healthy hand 0.2 deg and the glove 73.8. A joint limit
is drive-independent -- the fingertip simply cannot curl past it, whatever is
pulling. Applied to every condition equally.
"""
import sys, os, numpy as np, mujoco, collections
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
DIP=['md2_flexion','md3_flexion','md4_flexion','md5_flexion']
SEG={'proximal':{'proxph2','proxph3','proxph4','proxph5','proximal_thumb'},
     'middle':{'midph2','midph3','midph4','midph5'},
     'distal':{'distph2','distph3','distph4','distph5','distal_thumb'},
     'palm':{'capitate','hamate','trapezoid','trapezium','secondmc','thirdmc','fourthmc','fifthmc'}}
_orig=hb.GloveDevAdapter.build
def patched(self,rng,mass_override=None):
    m,d=_orig(self,rng,mass_override=mass_override)
    if hb._DIPMAX is not None:
        for j in DIP:
            try: jid=m.joint(j).id
            except KeyError: continue
            lo,hi=m.jnt_range[jid]
            m.jnt_range[jid]=[lo, min(hi, np.radians(hb._DIPMAX))]
    return m,d
hb.GloveDevAdapter.build=patched
print("index DIP at closure, load on tips, and survival   (natural DIP 10-40 deg)\n")
for cap in (None, 45.0, 30.0, 20.0):
    hb._DIPMAX=cap
    out=[]
    for cond in ('healthy_box','glove_dev_box'):
        ad=hb.glove_dev_adapters()[cond]
        m,d=ad.build(np.random.default_rng(2000)); mujoco.mj_forward(m,d)
        hb.settle(m,d,ad.oid,ad.settle_cap)
        ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
        for s_ in range(int(4.0/m.opt.timestep)):
            ad.set_input(m,d,min(s_/ramp,1.0)); mujoco.mj_step(m,d)
        seg=collections.Counter(); ft=np.zeros(6)
        for i in range(d.ncon):
            c=d.contact[i]
            if c.geom1 in ad.obj_geoms: o=m.geom_bodyid[c.geom2]
            elif c.geom2 in ad.obj_geoms: o=m.geom_bodyid[c.geom1]
            else: continue
            bn=m.body(o).name; mujoco.mj_contactForce(m,d,i,ft)
            for sg,n in SEG.items():
                if bn in n: seg[sg]+=abs(float(ft[0]))
        tot=sum(seg.values()) or 1
        dip=np.degrees(float(d.qpos[m.joint('md2_flexion').qposadr[0]]))
        st=hb.summarise([hb.run_trial(hb.glove_dev_adapters()[cond],2000+i,10.0) for i in range(6)])
        out.append(f"{cond.split('_')[0][:7]:7} DIP{dip:6.1f} tips{100*seg.get('distal',0)/tot:3.0f}% "
                   f"palm{100*seg.get('palm',0)/tot:3.0f}% surv{st['survival_rate']*100:4.0f}%")
    lab = "none" if cap is None else f"{cap:.0f} deg"
    print(f"  DIP cap {lab:8}  " + "   |   ".join(out), flush=True)
