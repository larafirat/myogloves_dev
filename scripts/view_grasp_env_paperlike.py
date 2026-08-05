"""Viewer for the separate paper-like D4_v2 thumb branch.

Usage:
    mjpython myogloves_dev/scripts/view_grasp_env_paperlike.py [DEVICE_NAME] [OBJECT_NAME]
"""

import sys
import time

import mujoco.viewer
import numpy as np

from grasp_env import GraspEnv, PAPERLIKE_MODEL_PATH
from run_grasp_baseline import OBJECT_VARIANTS, configure_object_variant

device_name = sys.argv[1] if len(sys.argv) > 1 else "D4_v2_paper_like_thumb_calibrated"
object_name = sys.argv[2] if len(sys.argv) > 2 else "tall_box"
env = GraspEnv(device_name, seed=0, model_path=PAPERLIKE_MODEL_PATH)
obs = env.reset(randomize=False)
variant = next(v for v in OBJECT_VARIANTS if v["name"] == object_name)
configure_object_variant(env.model, env.data, variant)

print(f"Viewing {device_name} in paper-like GraspEnv with {object_name} "
      f"(size~{env.model.geom_size[env.obj_geom_id]}, mass~{env.model.body_mass[env.obj_body_id]:.4f}kg)")

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
