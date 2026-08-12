"""Which digits actually touch, and where is the index relative to the box?"""
import sys, os, numpy as np, mujoco, collections
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
DIG={'thumb':{'proximal_thumb','distal_thumb'},'index':{'proxph2','midph2','distph2'},
     'middle':{'proxph3','midph3','distph3'},'ring':{'proxph4','midph4','distph4'},
     'little':{'proxph5','midph5','distph5'},'palm':{'secondmc','thirdmc','fourthmc','fifthmc','firstmc'}}
for cond in ('healthy_box','glove_dev_box'):
    ad=hb.glove_dev_adapters()[cond]
    m,d=ad.build(np.random.default_rng(2000)); mujoco.mj_forward(m,d)
    hb.settle(m,d,ad.oid,ad.settle_cap)
    q0=d.xquat[ad.oid].copy(); p0=d.xpos[ad.oid].copy()
    ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
    for k in range(int(4.0/m.opt.timestep)):
        ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    per=collections.Counter(); ft=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i]
        if c.geom1 in ad.obj_geoms: o=m.geom_bodyid[c.geom2]
        elif c.geom2 in ad.obj_geoms: o=m.geom_bodyid[c.geom1]
        else: continue
        bn=m.body(o).name; mujoco.mj_contactForce(m,d,i,ft)
        for dg,names in DIG.items():
            if bn in names: per[dg]+=abs(float(ft[0]))
    # object tilt since settle
    qa,qb=q0,d.xquat[ad.oid]
    dot=abs(float(np.dot(qa,qb))); tilt=np.degrees(2*np.arccos(min(dot,1.0)))
    print(f"\n=== {cond}")
    print("   force per digit (N): " + "  ".join(f"{k}={per.get(k,0):.0f}" for k in
          ('thumb','index','middle','ring','little','palm')))
    print(f"   object tilted {tilt:5.1f} deg and moved {np.linalg.norm(d.xpos[ad.oid]-p0)*1000:5.1f} mm since settle")
    # where is the index tip vs the box surface?
    tip=d.xpos[m.body('distph2').id]; obj=d.xpos[ad.oid]
    R=d.xmat[ad.oid].reshape(3,3); loc=R.T@(tip-obj)
    print(f"   index tip in the BOX's own frame: {np.round(loc*1000,1)} mm   (box half-extents 36 x 44 x 14 mm)")
    inside=all(abs(loc[i]*1000)<=(36,44,14)[i]+12 for i in range(3))
    print(f"   -> index tip is {'ON the box' if inside else 'OFF the box (past its surface)'}")
