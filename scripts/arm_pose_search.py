"""Which arm rotation presents the palm side-on, for a standing can?"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.STABILISE_WRIST=True
print(f"{'ARRx':>6}{'ARRy':>6}{'ARRz':>6}   {'palm pos':>22}{'palm->grasp dir':>22}  horiz?")
for arrx in (0.0, 0.6, 1.0, 1.4):
    for arrz in (0.0, -0.6, 0.6):
        ad=hb.glove_dev_adapters()['healthy_can']
        ad.arm_pose_override={'ARRx':arrx,'ARRz':arrz}
        ad.pos_override=(0.6,0.0,0.05)          # object parked away
        m,d=ad.build(np.random.default_rng(2000)); mujoco.mj_forward(m,d)
        hb.settle(m,d,ad.oid,ad.settle_cap)
        palm=d.xpos[m.body('capitate').id].copy()
        ramp=max(1,int(1.5/m.opt.timestep))
        for k in range(int(3.0/m.opt.timestep)):
            ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
        tips=np.array([d.xpos[m.body(b).id] for b in
                       ('distph2','distph3','distph4','distph5','distal_thumb')])
        c=tips.mean(axis=0); v=c-palm; v=v/np.linalg.norm(v)
        horiz = abs(v[2])<0.45     # palm looking sideways rather than down
        print(f"{arrx:6.1f}{0.0:6.1f}{arrz:6.1f}   {str(np.round(palm,3)):>22}"
              f"{str(np.round(v,2)):>22}  {'YES' if horiz else '-'}", flush=True)
