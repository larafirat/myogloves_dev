"""How far is the physics-derived grip target from what these grasps use?

F_target = m*g/mu * SAFETY is the classical "hold just above slip" setpoint.
If the simulated grasps need vastly more than that, they are not holding by
efficient force closure -- they are holding by crushing, and the setpoint is
measuring the wrong quantity.
"""
import sys, os, numpy as np, mujoco
sys.path.insert(0,'/Users/larafirat/myoassist/myogloves_dev/scripts'); os.chdir('/Users/larafirat/myoassist')
import hold_benchmark as hb

# what does an OPEN-LOOP grasp actually apply?
hb.FORCE_EASE_OFF=False
for cond,mass in (('glove_dev_can',0.349),('portOP_D4_v2_hybrid_per_finger_calibrated_can',0.349),
                  ('glove_dev_box',0.097)):
    ad=hb.glove_dev_adapters()[cond]
    m,d=ad.build(np.random.default_rng(1000)); mujoco.mj_forward(m,d)
    hb.settle(m,d,ad.oid,ad.settle_cap)
    reg=hb.GripForceRegulator(m,ad,mass)
    ramp=max(1,int(ad.closing_seconds/m.opt.timestep)); peak=0.0
    for k in range(int(4.0/m.opt.timestep)):
        ad.set_input(m,d,min(k/ramp,1.0)); mujoco.mj_step(m,d)
        peak=max(peak,reg.measure(m,d))
    print(f"{cond[:38]:40} target {reg.target:6.2f} N   actual peak {peak:8.1f} N   "
          f"ratio {peak/reg.target:6.0f}x")

print("\nsweeping the safety margin (can, glove_dev, 8 trials):")
hb.FORCE_EASE_OFF=True
for mar in (1.4, 5, 20, 50, 100):
    hb.GRIP_SAFETY_MARGIN=mar
    res=[hb.run_trial(hb.glove_dev_adapters()['glove_dev_can'],1000+i,5.0) for i in range(8)]
    s=hb.summarise(res)
    print(f"   margin {mar:5.1f} -> target {0.349*9.81*mar:6.1f} N   "
          f"surv={s['survival_rate']*100:5.0f}% grasp={s['grasp_rate']*100:5.0f}% hold={s['hold_median']:5.2f}s", flush=True)
