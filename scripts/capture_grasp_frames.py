"""Offscreen frame capture for visually self-checking a grasp run without a
live viewer window -- built after two headless-metric "improvements" in a row
turned out to look wrong live (repositioning the object based on contact-
timing numbers alone, twice). Renders PNGs at fixed moments (start, first
contact, settled grip, retract, post-retract) so changes can be checked by
actually looking at the frames before ever touching the live viewer again.

Usage:
    python myogloves_dev/scripts/capture_grasp_frames.py [device_name] [object_name] [out_dir]
"""

from __future__ import annotations

import sys
import os

import mujoco
import numpy as np

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController
from grasp_env import MODEL_PATH
from run_grasp_baseline import (
    OBJECT_VARIANTS,
    HAND_BODY_NAMES,
    RETRACT_TOUCH_STEPS,
    POST_CONTACT_SETTLE_STEPS,
    RETRACT_OFFSET,
    configure_object_variant,
)


def capture(device_name, object_name, out_dir, width=640, height=480):
    os.makedirs(out_dir, exist_ok=True)
    variant = next(v for v in OBJECT_VARIANTS if v["name"] == object_name)
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    configure_object_variant(model, data, variant)

    obj_geom_id = model.geom("grasp_object_geom").id
    start_pillar_body = model.body("start_pillar")
    start_pillar_id = start_pillar_body.id
    start_pillar_mocap_id = start_pillar_body.mocapid[0]
    hand_body_ids = {model.body(name).id for name in HAND_BODY_NAMES}

    device = DEVICES[device_name]
    if not device.tau_max_is_calibrated:
        device.tau_max[:] = 0.03
    applicator = ExoApplicator(model)
    controller = GraspController(model, device, obj_geom_id, **device.controller_overrides)

    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat = data.xpos[model.body("secondmc").id].copy()
    cam.distance = 0.45
    cam.azimuth = 130
    cam.elevation = -20

    def snap(name):
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        import PIL.Image
        path = os.path.join(out_dir, f"{name}.png")
        PIL.Image.fromarray(pixels).save(path)
        print(f"  saved {path}")

    snap("00_start")

    peak_touch_steps = 0
    settle_start_step = None
    retract_step = None
    contact_snapped = False
    settled_snapped = False
    original_pillar_pos = model.body_pos[start_pillar_id].copy()

    for step in range(6000):
        u = controller.step(model, data)
        applicator.apply(data, device, u=u, row_gate=controller.row_gate)
        mujoco.mj_step(model, data)

        touch = False
        for i in range(data.ncon):
            c = data.contact[i]
            if obj_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == obj_geom_id else c.geom1
            if model.geom_bodyid[other] in hand_body_ids:
                touch = True
        if touch:
            peak_touch_steps += 1
        else:
            peak_touch_steps = 0

        if touch and not contact_snapped:
            snap("01_first_contact")
            contact_snapped = True

        if retract_step is None and peak_touch_steps >= RETRACT_TOUCH_STEPS:
            if settle_start_step is None:
                settle_start_step = step
            if not settled_snapped and step - settle_start_step >= POST_CONTACT_SETTLE_STEPS // 2:
                snap("02_settling")
                settled_snapped = True
            if step - settle_start_step >= POST_CONTACT_SETTLE_STEPS:
                snap("03_pre_retract")
                data.mocap_pos[start_pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
                retract_step = step

        if retract_step is not None and step - retract_step == 250:
            snap("04_post_retract_0.5s")
        if retract_step is not None and step - retract_step == 1000:
            snap("05_post_retract_2s")
            break

    if retract_step is None:
        snap("99_never_contacted_end")


if __name__ == "__main__":
    device_name = sys.argv[1] if len(sys.argv) > 1 else "D4_v2_hybrid_per_finger_calibrated"
    object_name = sys.argv[2] if len(sys.argv) > 2 else "tall_box"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/tmp/grasp_frames"
    capture(device_name, object_name, out_dir)
