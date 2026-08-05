"""Visual runner for the retract and lift baseline tests.

Usage examples:
    mjpython myogloves_dev/scripts/view_baseline_test.py retract D4_v2_hybrid_per_finger_calibrated tall_box
    mjpython myogloves_dev/scripts/view_baseline_test.py lift D4_v2_hybrid_per_finger_calibrated tall_box
    mjpython myogloves_dev/scripts/view_baseline_test.py lift D4_v2_hybrid_per_finger_calibrated tall_box low_dip
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController
from grasp_env import MODEL_PATH, PAPERLIKE_MODEL_PATH
from run_grasp_baseline import (
    OBJECT_VARIANTS,
    HAND_BODY_NAMES,
    BODY_TO_DIGIT,
    POST_CONTACT_SETTLE_STEPS,
    RETRACT_TOUCH_STEPS,
    RETRACT_OFFSET,
    configure_object_variant,
)
from run_lift_baseline import LIFT_TOUCH_STEPS, LIFT_DROP


TEST_PROFILES = {
    "standard": {},
    "low_dip": {
        "mcp_scale": 0.92,
        "pip_lead": 0.10,
        "pip_full": 0.32,
        "dip_lead": 0.42,
        "dip_full": 0.82,
        "dip_scale": 0.55,
    },
}


def model_path_for_condition(condition_name: str) -> str:
    if condition_name == "D4_v2_paper_like_thumb_calibrated":
        return PAPERLIKE_MODEL_PATH
    return MODEL_PATH


def main():
    test_mode = sys.argv[1] if len(sys.argv) > 1 else "retract"
    condition_name = sys.argv[2] if len(sys.argv) > 2 else "D4_v2_hybrid_per_finger_calibrated"
    object_name = sys.argv[3] if len(sys.argv) > 3 else "tall_box"
    profile_name = sys.argv[4] if len(sys.argv) > 4 else "standard"

    if test_mode not in {"retract", "lift"}:
        raise SystemExit("test_mode must be 'retract' or 'lift'")
    if profile_name not in TEST_PROFILES:
        raise SystemExit(f"profile must be one of: {', '.join(TEST_PROFILES)}")

    variant = next(v for v in OBJECT_VARIANTS if v["name"] == object_name)
    model = mujoco.MjModel.from_xml_path(model_path_for_condition(condition_name))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configure_object_variant(model, data, variant)

    obj_geom_id = model.geom("grasp_object_geom").id
    pillar_geom_id = model.geom("start_pillar_geom").id
    if test_mode == "retract":
        model.geom_rgba[obj_geom_id] = np.array([0.20, 0.45, 0.95, 1.0])
        model.geom_rgba[pillar_geom_id] = np.array([0.18, 0.22, 0.40, 1.0])
        label = "RETRACT TEST"
    else:
        model.geom_rgba[obj_geom_id] = np.array([0.95, 0.70, 0.20, 1.0])
        model.geom_rgba[pillar_geom_id] = np.array([0.35, 0.28, 0.12, 1.0])
        label = "LIFT TEST"

    obj_body_id = model.body("grasp_object").id
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
        controller_kwargs = dict(device.controller_overrides)
        controller_kwargs.update(TEST_PROFILES[profile_name])
        controller = GraspController(model, device, obj_geom_id, **controller_kwargs)

    first_hand_touch_step = None
    sustained_touch_steps = 0
    peak_hand_touch_steps = 0
    touch_digits = set()
    event_step = None
    settle_start_step = None
    original_pillar_pos = model.body_pos[start_pillar_id].copy()
    event_touch_steps = RETRACT_TOUCH_STEPS if test_mode == "retract" else LIFT_TOUCH_STEPS

    print("=" * 72)
    print(f"{label}: {condition_name} with {object_name} [{profile_name}]")
    print("=" * 72)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if controller is not None:
                u = controller.step(model, data)
                applicator.apply(data, device, u=u, row_gate=controller.row_gate)
            mujoco.mj_step(model, data)

            hand_touch = False
            for i in range(data.ncon):
                c = data.contact[i]
                if obj_geom_id not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == obj_geom_id else c.geom1
                other_body = model.geom_bodyid[other]
                if other_body in hand_body_ids:
                    hand_touch = True
                    touch_digits.add(BODY_TO_DIGIT[model.body(other_body).name])

            if hand_touch:
                if first_hand_touch_step is None:
                    first_hand_touch_step = step
                sustained_touch_steps += 1
                peak_hand_touch_steps = max(peak_hand_touch_steps, sustained_touch_steps)
            else:
                sustained_touch_steps = 0
                # NOT resetting settle_start_step here: the trigger condition below
                # (controller.locked or peak_hand_touch_steps) is monotonic/sticky once
                # reached, so a momentary single-step contact gap shouldn't restart the
                # settle countdown from zero.

            # Wait for the FULL grasp (every channel locked), not just any sustained
            # contact -- the old touch-duration-only trigger could pull the support
            # before some digits (e.g. the extra thumb) had engaged at all.
            full_grasp = all(controller.locked) if controller is not None else (peak_hand_touch_steps >= event_touch_steps)
            if event_step is None and full_grasp:
                if settle_start_step is None:
                    settle_start_step = step
                    print(f"{label} contact threshold reached at t={step * model.opt.timestep:.3f}s; settling...")
                if step - settle_start_step >= POST_CONTACT_SETTLE_STEPS:
                    if test_mode == "retract":
                        data.mocap_pos[start_pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
                    else:
                        data.mocap_pos[start_pillar_mocap_id] = original_pillar_pos + LIFT_DROP
                    event_step = step
                    print(f"{label} triggered at t={step * model.opt.timestep:.3f}s; digits={','.join(sorted(touch_digits)) or 'none'}")

            viewer.sync()
            time.sleep(0.002)
            step += 1

            if step % 500 == 0:
                obj_pos = data.xpos[obj_body_id]
                print(
                    f"t={step * model.opt.timestep:.2f}s  "
                    f"obj=({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f})  "
                    f"peak_touch={peak_hand_touch_steps * model.opt.timestep:.3f}s  "
                    f"event={'yes' if event_step is not None else 'no'}  "
                    f"label={label}"
                )


if __name__ == "__main__":
    main()
