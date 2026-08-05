"""Simple lift-style benchmark across key conditions.

After sustained contact is established, the support pillar is lowered rather
than translated away. This is a closer proxy to "grab and lift" than the
existing retract test, while still working with the fixed-arm setup.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController
from grasp_env import MODEL_PATH, PAPERLIKE_MODEL_PATH

MAX_STEPS = 5000
LIFT_TOUCH_STEPS = 250
POST_CONTACT_SETTLE_STEPS = 800
POST_LIFT_STEPS = 1000
LIFT_DROP = np.array([0.0, 0.0, -0.20])
HEIGHT_SUCCESS_MARGIN = -0.03

CONDITIONS = [
    ("bare_msk", MODEL_PATH),
    ("D4_v2_hybrid_per_finger_calibrated", MODEL_PATH),
    ("D4_v2_paper_like_thumb_calibrated", PAPERLIKE_MODEL_PATH),
]

HAND_BODY_NAMES = [
    "firstmc", "proximal_thumb", "distal_thumb",
    "proxph2", "midph2", "distph2",
    "proxph3", "midph3", "distph3",
    "proxph4", "midph4", "distph4",
    "proxph5", "midph5", "distph5",
]

BODY_TO_DIGIT = {
    "firstmc": "thumb",
    "proximal_thumb": "thumb",
    "distal_thumb": "thumb",
    "proxph2": "index",
    "midph2": "index",
    "distph2": "index",
    "proxph3": "middle",
    "midph3": "middle",
    "distph3": "middle",
    "proxph4": "ring",
    "midph4": "ring",
    "distph4": "ring",
    "proxph5": "little",
    "midph5": "little",
    "distph5": "little",
}


def run_condition(condition_name, model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj_body_id = model.body("grasp_object").id
    obj_geom_id = model.geom("grasp_object_geom").id
    start_pillar_body = model.body("start_pillar")
    start_pillar_id = start_pillar_body.id
    start_pillar_mocap_id = start_pillar_body.mocapid[0]
    hand_body_ids = {model.body(name).id for name in HAND_BODY_NAMES}

    applicator = ExoApplicator(model)
    controller = None
    device = None
    if condition_name != "bare_msk":
        device = DEVICES[condition_name]
        if not device.tau_max_is_calibrated:
            device.tau_max[:] = 0.03
        controller = GraspController(model, device, obj_geom_id, **device.controller_overrides)

    first_hand_touch_step = None
    sustained_touch_steps = 0
    peak_hand_touch_steps = 0
    touch_digits = set()
    lift_step = None
    z_at_lift = None
    settle_start_step = None
    original_pillar_pos = model.body_pos[start_pillar_id].copy()

    for step in range(MAX_STEPS):
        if controller is not None:
            u = controller.step(model, data)
            applicator.apply(data, device, u=u, row_gate=controller.row_gate)
        mujoco.mj_step(model, data)

        hand_touch = False
        step_touch_digits = set()
        for i in range(data.ncon):
            c = data.contact[i]
            if obj_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == obj_geom_id else c.geom1
            other_body = model.geom_bodyid[other]
            if other_body in hand_body_ids:
                hand_touch = True
                step_touch_digits.add(BODY_TO_DIGIT[model.body(other_body).name])

        if hand_touch:
            if first_hand_touch_step is None:
                first_hand_touch_step = step
            sustained_touch_steps += 1
            peak_hand_touch_steps = max(peak_hand_touch_steps, sustained_touch_steps)
            if lift_step is None:
                touch_digits.update(step_touch_digits)
        else:
            sustained_touch_steps = 0
            settle_start_step = None

        if lift_step is None and peak_hand_touch_steps >= LIFT_TOUCH_STEPS:
            if settle_start_step is None:
                settle_start_step = step
            if step - settle_start_step >= POST_CONTACT_SETTLE_STEPS:
                data.mocap_pos[start_pillar_mocap_id] = original_pillar_pos + LIFT_DROP
                lift_step = step
                z_at_lift = float(data.xpos[obj_body_id][2])

        if lift_step is not None and step - lift_step >= POST_LIFT_STEPS:
            break

    z_drop = None
    held = None
    if z_at_lift is not None:
        z_drop = float(data.xpos[obj_body_id][2] - z_at_lift)
        held = z_drop > HEIGHT_SUCCESS_MARGIN

    return {
        "condition": condition_name,
        "contact_time_s": None if first_hand_touch_step is None else first_hand_touch_step * model.opt.timestep,
        "peak_sustained_touch_s": peak_hand_touch_steps * model.opt.timestep,
        "lift_time_s": None if lift_step is None else lift_step * model.opt.timestep,
        "held_after_lift": held,
        "z_change_after_lift_m": z_drop,
        "touch_digits": ",".join(sorted(touch_digits)) if touch_digits else "none",
    }


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.3f}"


def main():
    print("Condition                          Contact (s)  Peak touch (s)  Lift (s)  Held?  dZ after lift (m)  Touch digits")
    print("---------------------------------------------------------------------------------------------------------------")
    for condition_name, model_path in CONDITIONS:
        result = run_condition(condition_name, model_path)
        print(
            f"{result['condition']:<33}  "
            f"{fmt(result['contact_time_s']):>11}  "
            f"{fmt(result['peak_sustained_touch_s']):>14}  "
            f"{fmt(result['lift_time_s']):>8}  "
            f"{str(result['held_after_lift']):>5}  "
            f"{fmt(result['z_change_after_lift_m']):>17}  "
            f"{result['touch_digits']}"
        )


if __name__ == "__main__":
    main()
