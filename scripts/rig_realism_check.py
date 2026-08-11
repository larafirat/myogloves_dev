"""Two rig-realism checks: is the wrist floppy, and is the object out of reach
of the palm?"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

WRIST=["pro_sup","deviation","flexion"]
m0=mujoco.MjModel.from_xml_path('myogloves_dev/models/myohand_glove_dev.xml')
print("=== 1. IS THE WRIST DRIVEN BY ANYTHING? ===")
acts=[m0.actuator(i).name for i in range(m0.nu)]
print("   actuators in model:", acts)
for j in WRIST:
    jid=m0.joint(j).id; dof=m0.jnt_dofadr[j if isinstance(j,int) else m0.joint(j).id]
    driven=[a for a in acts if a.lower().startswith(j[:4].lower())]
    print(f"   {j:10} damping={m0.dof_damping[m0.joint(j).dofadr[0]]:.3f}  "
          f"range={np.degrees(m0.jnt_range[jid]).round(0)}  actuator={driven or 'NONE'}")

print("\n=== 2. WRIST MOTION DURING A TRIAL (glove_dev/can) ===")
ad=hb.glove_dev_adapters()['glove_dev_can']
m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
q0={j:float(d.qpos[m.joint(j).qposadr[0]]) for j in WRIST}
hb.settle(m,d,ad.oid,ad.settle_cap)
qs={j:float(d.qpos[m.joint(j).qposadr[0]]) for j in WRIST}
print("   joint      start->settled     then during close+hold")
ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
ext={j:[qs[j],qs[j]] for j in WRIST}
for k in range(int(6.0/m.opt.timestep)):
    ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
    for j in WRIST:
        v=float(d.qpos[m.joint(j).qposadr[0]]); ext[j][0]=min(ext[j][0],v); ext[j][1]=max(ext[j][1],v)
for j in WRIST:
    print(f"   {j:10} {np.degrees(q0[j]):+6.1f} -> {np.degrees(qs[j]):+6.1f} deg   "
          f"then swings {np.degrees(ext[j][1]-ext[j][0]):5.1f} deg")

print("\n=== 3. HOW FAR IS THE OBJECT FROM THE PALM AT START? ===")
for cond,r in (('glove_dev_box',0.036),('glove_dev_can',0.033),('glove_dev_tuna',0.042)):
    ad=hb.glove_dev_adapters()[cond]
    m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
    hb.settle(m,d,ad.oid,ad.settle_cap)
    palm=float(np.linalg.norm(d.xpos[ad.oid]-d.xpos[m.body('capitate').id]))
    # fingertip distance for comparison
    tip=float(np.linalg.norm(d.xpos[ad.oid]-d.xpos[m.body('distph3').id]))
    print(f"   {cond:16} object->palm {palm*1000:5.1f} mm   object->middle fingertip {tip*1000:5.1f} mm"
          f"   (object radius {r*1000:.0f} mm)")
    print(f"   {'':16} palm CLEARANCE past the object surface: {(palm-r)*1000:5.1f} mm")
