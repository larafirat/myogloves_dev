"""Live viewer for GraspEnv (myohand_exoglove_env.xml) -- the MyoChallenge-
style structured environment, separate from view_grasp_test.py's ad hoc one.
Prints the obs/reward dict periodically so you can see reach_err shrink and
touching_body/reward change live alongside the visual.

Usage (must run with mjpython on macOS for the interactive window):
    mjpython myogloves_dev/scripts/view_grasp_env.py [DEVICE_NAME]

Defaults to D4_hybrid_per_finger_calibrated if no device name is given.
"""

import sys
import time

import mujoco.viewer
import numpy as np

from grasp_env import GraspEnv

device_name = sys.argv[1] if len(sys.argv) > 1 else "D4_hybrid_per_finger_calibrated"
env = GraspEnv(device_name, seed=0)
obs = env.reset(randomize=False)

print(f"Viewing {device_name} in GraspEnv (randomized object: "
      f"size~{env.model.geom_size[env.obj_geom_id]}, mass~{env.model.body_mass[env.obj_body_id]:.4f}kg)")

step_count = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        obs, rew = env.step()
        viewer.sync()
        time.sleep(0.002)
        step_count += 1
        if step_count % 500 == 0:
            print(f"t={obs['time'][0]:.2f}s  reach_err={np.linalg.norm(obs['reach_err']):.4f}  "
                  f"touching(hand/start/other)={obs['touching_body']}  "
                  f"reward(reach/wrap/total)={rew['reach']:.3f}/{rew['wrap']:.3f}/{rew['total']:.3f}  "
                  f"success={env.success()}")
