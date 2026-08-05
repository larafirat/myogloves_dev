"""Viewer for separate grasp-shape experiments.

Keeps the current main D4_v2 setup untouched while letting us try a closer
object placement and wrap-first controller settings in a separate scene.

Usage:
    mjpython myogloves_dev/scripts/view_grasp_experiment.py [DEVICE_NAME]
"""

import sys
import time

import mujoco.viewer
import numpy as np

from grasp_env import GraspEnv

MODEL_PATH = "myogloves_dev/models/myohand_exoglove_env_grasp_experiment.xml"

device_name = sys.argv[1] if len(sys.argv) > 1 else "D4_v2_hybrid_per_finger_calibrated"

controller_overrides = {
    # Let MCP establish contact first, then bring PIP in more gradually so
    # the object is wrapped instead of flicked outward when the middle joints engage.
    "mcp_scale": 0.88,
    "pip_lead": 0.10,
    "pip_full": 0.30,
    "dip_lead": 0.18,
    "dip_full": 0.48,
    "dip_scale": 0.50,
    # Let the thumb opposition complete the grasp instead of hanging back so
    # long that the thumb IP never comes into real contact with the index side.
    "abduct_lead": 0.26,
    "abduct_full": 0.70,
}

env = GraspEnv(
    device_name,
    seed=0,
    model_path=MODEL_PATH,
    controller_overrides=controller_overrides,
)
obs = env.reset(randomize=False)
obj_qadr = env.obj_qposadr
obj_vadr = env.obj_dofadr
obj_quat0 = env.data.qpos[obj_qadr + 3:obj_qadr + 7].copy()

# Local-only squeeze boost for the grasp experiment: increase the active finger
# channels a bit and give the thumb a modest extra finish for full opposition.
env.device.tau_max[0] *= 1.40  # index
env.device.tau_max[1] *= 1.36  # middle
env.device.tau_max[2] *= 1.18  # thumb
env.device.tau_max[3] *= 1.32  # shared ring+pinky

print(
    f"Viewing grasp experiment for {device_name} "
    f"(size~{env.model.geom_size[env.obj_geom_id]}, "
    f"mass~{env.model.body_mass[env.obj_body_id]:.4f}kg)"
)

step_count = 0
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        obs, rew = env.step()
        # Grasp-experiment simplification only: keep the object from rotating so
        # we can judge hand closure on a smaller box without the free body
        # twisting out of the grasp.
        env.data.qpos[obj_qadr + 3:obj_qadr + 7] = obj_quat0
        env.data.qvel[obj_vadr + 3:obj_vadr + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)
        viewer.sync()
        time.sleep(0.002)
        step_count += 1
        if step_count % 500 == 0:
            print(
                f"t={obs['time'][0]:.2f}s  reach_err={np.linalg.norm(obs['reach_err']):.4f}  "
                f"touching(hand/start/other)={obs['touching_body']}  "
                f"reward(reach/wrap/total)={rew['reach']:.3f}/{rew['wrap']:.3f}/{rew['total']:.3f}  "
                f"u={np.round(obs['u'], 3)}"
            )
