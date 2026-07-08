"""The real test: once a device's closed-loop grip (grasp_controller.py) locks
onto the object, RETRACT the fixed support out from under it and check whether
the hand alone keeps it up against gravity. Earlier versions of this test left
the support in place the whole time, so "contained" only ever meant "didn't
get knocked off a platform it was already resting on stably" -- that passes
even when the hand is doing none of the load-bearing work, which is not a
meaningful test of grasping.

Uses models/myohand_exoglove_grasp.xml: a YCB-benchmark box (Calli et al.
2015; 97g, 72x88x28mm) on a small MOCAP (kinematic, retractable at runtime)
support, positioned at the real tripod-pinch convergence point of D4's
index/middle/thumb fingertips (traced by driving all three simultaneously --
see the model file's comments).

SCOPE NOTE: D4 (Gerez et al. 2020) is the only device here with a real,
independently-actuated thumb (n_inputs=3). D1-D3's papers only ever published
index+middle data, so they drive 2 digits at this same spot with nothing
opposing them -- without the support to lean on, expect them to fail this
test outright, since two fingers pushing in roughly the same direction cannot
suspend an object against gravity with nothing on the other side.

Run from the repo root: python myogloves_dev/scripts/grasp_test.py
"""

import mujoco
import numpy as np

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController

MODEL_PATH = "myogloves_dev/models/myohand_exoglove_grasp.xml"
MAX_STEPS = 10000       # 20 sim-seconds total, ceiling for the whole run
POST_RETRACT_STEPS = 3000  # ~6s given after retraction to see if it actually holds
RETRACT_OFFSET = np.array([1.0, 0.0, 0.0])  # move the mocap support sideways, out of the object's
# footprint entirely -- moving it straight down (an earlier attempt) kept the same x/y, so it just
# became a lower table the box fell onto and landed on again: drift was exactly the retract distance
# for every device, a dead giveaway that "support" was still catching it, not actually gone.
HELD_TOL = 0.02  # m; must stay within this of its position-at-retraction to count as "held"

ORDER = [
    "D1_underactuated_distal", "D1_underactuated_distal_calibrated",
    "D2_synergy_cross_finger", "D2_synergy_cross_finger_calibrated",
    "D3_uniform_single_dof", "D3_uniform_single_dof_calibrated",
    "D4_hybrid_per_finger", "D4_hybrid_per_finger_calibrated",
]

IGNORE_GEOMS = {"grasp_support_geom", "floor"}


def run(device):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    applicator = ExoApplicator(model)
    if not device.tau_max_is_calibrated:
        device.tau_max[:] = 0.03  # same below-saturation convention as view_exo_device.py

    obj_id = model.body("grasp_object").id
    obj_geom_id = model.geom("grasp_object_geom").id
    support_mocap_id = model.body("grasp_support").mocapid[0]
    original_support_pos = model.body_pos[model.body("grasp_support").id].copy()
    mujoco.mj_forward(model, data)

    controller = GraspController(model, device, obj_geom_id)
    retract_step = None
    pos_at_retract = None

    for step in range(MAX_STEPS):
        u = controller.step(model, data)
        applicator.apply(data, device, u=u, row_gate=controller.row_gate)
        mujoco.mj_step(model, data)

        if retract_step is None and all(controller.locked):
            retract_step = step
            data.mocap_pos[support_mocap_id] = original_support_pos + RETRACT_OFFSET
            pos_at_retract = data.xpos[obj_id].copy()

        if retract_step is not None and step - retract_step >= POST_RETRACT_STEPS:
            break

    mujoco.mj_forward(model, data)
    end_pos = data.xpos[obj_id].copy()

    if retract_step is None:
        return {"gripped": False, "held": False, "grip_time_s": None, "drift_after_retract": None}

    drift = float(np.linalg.norm(end_pos - pos_at_retract))
    held = drift < HELD_TOL
    return {
        "gripped": True,
        "held": held,
        "grip_time_s": retract_step * 0.002,
        "drift_after_retract": drift,
    }


def main():
    cols = ("Device", "Gripped?", "Grip time (s)", "Held w/o support?", "Drift after retract (m)")
    widths = (36, 10, 15, 19, 24)
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for name in ORDER:
        device = DEVICES[name]
        r = run(device)
        grip_time = f"{r['grip_time_s']:.2f}" if r["grip_time_s"] is not None else "n/a"
        drift = f"{r['drift_after_retract']:.4f}" if r["drift_after_retract"] is not None else "n/a"
        vals = (name, "yes" if r["gripped"] else "NO", grip_time,
                "YES" if r["held"] else ("no" if r["gripped"] else "n/a"), drift)
        print("  ".join(str(v).ljust(w) for v, w in zip(vals, widths)))


if __name__ == "__main__":
    main()
