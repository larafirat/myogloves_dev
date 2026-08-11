"""Fingertip pinch or whole-hand wrap? Measures WHICH phalanx touches, and how
far each joint flexes.

A power grasp loads the proximal and middle phalanges and keeps the DIP
relatively straight. A pinch loads only the distal phalanges.
"""
import sys, os, numpy as np, mujoco, collections
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

SEG={'proximal':{'proxph2','proxph3','proxph4','proxph5','proximal_thumb'},
     'middle':  {'midph2','midph3','midph4','midph5'},
     'distal':  {'distph2','distph3','distph4','distph5','distal_thumb'},
     'palm':    {'capitate','hamate','trapezoid','trapezium','secondmc','thirdmc','fourthmc','fifthmc'}}
JT=[('index','mcp2_flexion','pm2_flexion','md2_flexion'),
    ('middle','mcp3_flexion','pm3_flexion','md3_flexion'),
    ('ring','mcp4_flexion','pm4_flexion','md4_flexion')]

for cond in ('healthy_box','glove_dev_box'):
    ad=hb.glove_dev_adapters()[cond]
    m,d=ad.build(np.random.default_rng(2000)); mujoco.mj_forward(m,d)
    hb.settle(m,d,ad.oid,ad.settle_cap)
    ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
    for k in range(int(4.0/m.opt.timestep)):
        ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    seg=collections.Counter(); ft=np.zeros(6); force=collections.Counter()
    for i in range(d.ncon):
        c=d.contact[i]
        if c.geom1 in ad.obj_geoms: o=m.geom_bodyid[c.geom2]
        elif c.geom2 in ad.obj_geoms: o=m.geom_bodyid[c.geom1]
        else: continue
        bn=m.body(o).name
        mujoco.mj_contactForce(m,d,i,ft)
        for s,names in SEG.items():
            if bn in names: seg[s]+=1; force[s]+=abs(float(ft[0]))
    print(f"\n=== {cond}")
    print("   contacts by segment:  " + "  ".join(
        f"{s}={seg.get(s,0)} ({force.get(s,0):.0f} N)" for s in ('palm','proximal','middle','distal')))
    print("   joint flexion at closure (deg):")
    for name,mcp,pip,dip in JT:
        v=[np.degrees(float(d.qpos[m.joint(j).qposadr[0]])) for j in (mcp,pip,dip)]
        print(f"      {name:7} MCP {v[0]:6.1f}   PIP {v[1]:6.1f}   DIP {v[2]:6.1f}")
    print("   natural power grasp for reference: MCP 60-90, PIP 60-90, DIP 10-40")
