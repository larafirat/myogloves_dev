"""Fast geometric screen: where can the object SIT undisturbed and then be
enclosed by the closing hand -- including palm contact?

No hold phase, so this is cheap. Reports, per candidate: did it survive
settling, how many digits touch at closure, and crucially whether the PALM
touches -- the thing no placement in this rig has ever achieved.
"""
import sys, os, itertools, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.STABILISE_WRIST=True
PALM_BODIES={'capitate','hamate','trapezoid','trapezium','secondmc','thirdmc','fourthmc','fifthmc'}

def probe(obj, pos, seed=2000):
    ad=hb.glove_dev_adapters()[f'healthy_{obj}']; ad.pos_override=pos
    m,d=ad.build(np.random.default_rng(seed)); mujoco.mj_forward(m,d)
    z0=float(d.xpos[ad.oid][2])
    steps,ok=hb.settle(m,d,ad.oid,ad.settle_cap)
    if not ok: return None
    moved=float(np.linalg.norm(d.xpos[ad.oid]-np.array(pos)))
    ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
    for k in range(int(3.5/m.opt.timestep)):
        ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    digits=hb._digits_in_contact(m,d,ad)
    palm=False
    for i in range(d.ncon):
        c=d.contact[i]
        if c.geom1 in ad.obj_geoms: other=m.geom_bodyid[c.geom2]
        elif c.geom2 in ad.obj_geoms: other=m.geom_bodyid[c.geom1]
        else: continue
        if m.body(other).name in PALM_BODIES: palm=True
    return len(digits), sorted(digits), palm, moved

XS=[0.095,0.113,0.130]; YS=[0.020,0.035,0.050]; ZS=[0.145,0.160,0.175]
for obj in ('can','box'):
    print(f"\n=== {obj.upper()}   (settle-survivors only)")
    for z,x,y in itertools.product(ZS,XS,YS):
        r=probe(obj,(x,y,z))
        if r is None: continue
        n,digs,palm,moved=r
        if n>=2:
            print(f"   ({x},{y},{z})  {n} digits {str(digs):48} "
                  f"PALM={'YES' if palm else 'no '}  settled-drift {moved*1000:.0f}mm", flush=True)
