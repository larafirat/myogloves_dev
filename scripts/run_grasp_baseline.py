"""Baseline comparison harness across a small object set.

Compares bare MSK and selected exoskeleton conditions on the same widened
pedestal using a few stable analytic object shapes. This gives us a first
object-by-object answer to "is the exoskeleton helping?" before we move on to
full lift benchmarks.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController
from grasp_env import MODEL_PATH, PAPERLIKE_MODEL_PATH

MAX_STEPS = 5000
RETRACT_TOUCH_STEPS = 250
POST_CONTACT_SETTLE_STEPS = 800
POST_RETRACT_STEPS = 1000
RETRACT_OFFSET = np.array([1.0, 0.0, 0.0])
PILLAR_TOP_Z = 0.4715 + 0.8715

CONDITIONS = [
    ("bare_msk", MODEL_PATH),
    ("D4_v2_hybrid_per_finger_calibrated", MODEL_PATH),
    ("D4_v2_paper_like_thumb_calibrated", PAPERLIKE_MODEL_PATH),
]

OBJECT_VARIANTS = [
    {
        "name": "tall_box",
        "geom_type": mujoco.mjtGeom.mjGEOM_BOX,
        "size": np.array([0.014, 0.036, 0.044]),
        "mass": 0.097,
        "pos": np.array([-0.160, -0.490, PILLAR_TOP_Z + 0.044]),
    },
    {
        "name": "can_cylinder",
        "geom_type": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "size": np.array([0.025, 0.040, 0.0]),
        "mass": 0.097,
        "pos": np.array([-0.160, -0.490, PILLAR_TOP_Z + 0.040]),
    },
    {
        "name": "apple_sphere",
        "geom_type": mujoco.mjtGeom.mjGEOM_SPHERE,
        "size": np.array([0.032, 0.0, 0.0]),
        "mass": 0.097,
        "pos": np.array([-0.160, -0.490, PILLAR_TOP_Z + 0.032]),
    },
    {
        # Official MLB baseball: 74mm diameter (37mm radius), 145g.
        "name": "baseball",
        "geom_type": mujoco.mjtGeom.mjGEOM_SPHERE,
        "size": np.array([0.037, 0.0, 0.0]),
        "mass": 0.145,
        "pos": np.array([-0.160, -0.490, PILLAR_TOP_Z + 0.037]),
    },
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


def configure_object_variant(model, data, variant):
    obj_body_id = model.body("grasp_object").id
    obj_geom_id = model.geom("grasp_object_geom").id
    obj_joint_id = model.body_jntadr[obj_body_id]
    qadr = model.jnt_qposadr[obj_joint_id]
    vadr = model.jnt_dofadr[obj_joint_id]

    model.geom_type[obj_geom_id] = int(variant["geom_type"])
    model.geom_size[obj_geom_id] = variant["size"]
    model.body_mass[obj_body_id] = float(variant["mass"])

    data.qpos[qadr:qadr + 3] = variant["pos"]
    data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def run_condition(condition_name, model_path, variant):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configure_object_variant(model, data, variant)

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
    retract_step = None
    pos_at_retract = None
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
            if retract_step is None:
                touch_digits.update(step_touch_digits)
        else:
            sustained_touch_steps = 0
            # NOT resetting settle_start_step here anymore: both full_grasp conditions
            # below (controller.locked and peak_hand_touch_steps) are monotonic/sticky
            # once reached, so a momentary single-step contact gap shouldn't restart
            # the settle countdown from zero.

        # Retract once the FULL grasp is established (every channel locked), not just
        # once ANY contact has been sustained a while -- the old touch-duration-only
        # trigger could pull the support before some digits (e.g. the extra thumb)
        # had engaged at all, understating how well a device actually holds.
        full_grasp = all(controller.locked) if controller is not None else (peak_hand_touch_steps >= RETRACT_TOUCH_STEPS)
        if retract_step is None and full_grasp:
            if settle_start_step is None:
                settle_start_step = step
            if step - settle_start_step >= POST_CONTACT_SETTLE_STEPS:
                data.mocap_pos[start_pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
                retract_step = step
                pos_at_retract = data.xpos[obj_body_id].copy()

        if retract_step is not None and step - retract_step >= POST_RETRACT_STEPS:
            break

    drift = None
    held = None
    if pos_at_retract is not None:
        drift = float(np.linalg.norm(data.xpos[obj_body_id] - pos_at_retract))
        held = drift < 0.05

    return {
        "object": variant["name"],
        "condition": condition_name,
        "contact_time_s": None if first_hand_touch_step is None else first_hand_touch_step * model.opt.timestep,
        "peak_sustained_touch_s": peak_hand_touch_steps * model.opt.timestep,
        "touch_digits": ",".join(sorted(touch_digits)) if touch_digits else "none",
        "retract_time_s": None if retract_step is None else retract_step * model.opt.timestep,
        "held_after_retract": held,
        "drift_after_retract_m": drift,
    }


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return "n/a"
    return f"{v:.3f}"


def main():
    print("Object         Condition                          Contact (s)  Peak touch (s)  Retract (s)  Held?  Drift (m)  Touch digits")
    print("----------------------------------------------------------------------------------------------------------------------------")
    for variant in OBJECT_VARIANTS:
        for condition_name, model_path in CONDITIONS:
            result = run_condition(condition_name, model_path, variant)
            print(
                f"{result['object']:<13}"
                f"{result['condition']:<33}  "
                f"{fmt(result['contact_time_s']):>11}  "
                f"{fmt(result['peak_sustained_touch_s']):>14}  "
                f"{fmt(result['retract_time_s']):>11}  "
                f"{str(result['held_after_retract']):>5}  "
                f"{fmt(result['drift_after_retract_m']):>9}  "
                f"{result['touch_digits']}"
            )


if __name__ == "__main__":
    main()
