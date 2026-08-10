"""What grip margin does the HEALTHY HAND use?

If the reference condition holds at N x its slip threshold, then N is a
biologically-derived setpoint for the regulator -- taken from the hand the
devices are assisting, rather than fitted to whichever device it flatters.
"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb
hb.FORCE_EASE_OFF=False
print(f"{'condition':28}{'slip thresh':>12}{'peak applied':>14}{'ratio':>9}   (only trials that HELD)")
for cond,mass in (("healthy_box",0.097),("healthy_can",0.349),("healthy_tuna",0.171)):
    ratios=[]
    for seed in range(1000,1008):
        ad=hb.glove_dev_adapters()[cond]
        m,d=ad.build(np.random.default_rng(seed)); mujoco.mj_forward(m,d)
        if not hb.settle(m,d,ad.oid,ad.settle_cap)[1]: continue
        reg=hb.GripForceRegulator(m,ad,mass); reg.target=mass*9.81/1.0   # bare slip threshold
        ramp=max(1,int(ad.closing_seconds/m.opt.timestep))
        # close, then hold through the window and take the force while holding
        for k in range(ramp+int(2.0/m.opt.timestep)):
            ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
        ad.release_support(m,d); z0=float(d.xpos[ad.oid][2]); held=[]
        for k in range(int(3.0/m.opt.timestep)):
            ad.set_input(m,d,1.0); mujoco.mj_step(m,d)
            if z0-float(d.xpos[ad.oid][2])>hb.DROP_THRESHOLD_M: break
            held.append(reg.measure(m,d))
        if len(held)>500:            # only count trials that actually held
            ratios.append(float(np.median(held))/reg.target)
    if ratios:
        print(f"{cond:28}{mass*9.81:9.2f} N {np.median(ratios)*mass*9.81:11.2f} N "
              f"{np.median(ratios):8.1f}x   (n={len(ratios)})")
    else:
        print(f"{cond:28}{mass*9.81:9.2f} N {'no held trials':>14}")
