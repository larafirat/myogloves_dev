"""Is the fingertip curl caused by the object being too THIN?

The hand's measured enclosure is ~64 mm. The gelatin box is 28 mm across the
fingers, so they can close almost fully before meeting it -- ending in a fist
with the box caught in the fingertips. A thicker object should stop them
earlier and give a wrap.
"""
import sys, os, numpy as np, mujoco, collections
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
SEG={'proximal':{'proxph2','proxph3','proxph4','proxph5','proximal_thumb'},
     'middle':{'midph2','midph3','midph4','midph5'},
     'distal':{'distph2','distph3','distph4','distph5','distal_thumb'},
     'palm':{'capitate','hamate','trapezoid','trapezium','secondmc','thirdmc','fourthmc','fifthmc'}}
CASES=[('gelatin box','healthy_box',28,None),
       ('pudding box','healthy_pudding',36,None),
       ('potted meat','healthy_meat',52,(0.095,0.05,0.115))]
print("index angles + contact force by segment   (natural power grasp: DIP 10-40 deg)\n")
for label,cond,width,pos in CASES:
    ad=hb.glove_dev_adapters()[cond]
    if pos: ad.pos_override=pos
    m,d=ad.build(np.random.default_rng(2000)); mujoco.mj_forward(m,d)
    hb.settle(m,d,ad.oid,ad.settle_cap)
    ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
    for k in range(int(4.0/m.opt.timestep)):
        ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    seg=collections.Counter(); ft=np.zeros(6)
    for i in range(d.ncon):
        c=d.contact[i]
        if c.geom1 in ad.obj_geoms: o=m.geom_bodyid[c.geom2]
        elif c.geom2 in ad.obj_geoms: o=m.geom_bodyid[c.geom1]
        else: continue
        bn=m.body(o).name
        mujoco.mj_contactForce(m,d,i,ft)
        for s,names in SEG.items():
            if bn in names: seg[s]+=abs(float(ft[0]))
    a=[np.degrees(float(d.qpos[m.joint(j).qposadr[0]])) for j in ('mcp2_flexion','pm2_flexion','md2_flexion')]
    tot=sum(seg.values()) or 1
    print(f"{label:13} {width:3d}mm   MCP{a[0]:6.1f} PIP{a[1]:6.1f} DIP{a[2]:6.1f}   "
          f"palm{100*seg.get('palm',0)/tot:4.0f}% prox{100*seg.get('proximal',0)/tot:4.0f}% "
          f"mid{100*seg.get('middle',0)/tot:4.0f}% dist{100*seg.get('distal',0)/tot:4.0f}%", flush=True)
