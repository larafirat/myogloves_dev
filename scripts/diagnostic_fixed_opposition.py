"""Diagnostic: does the grasp fail from lack of OPPOSITION, or from wrong
contact geometry even when opposition exists?

Adapted to this project's actual setup (myohand_exoglove_env.xml, GraspEnv-
style retract/rotation diagnostic, GraspController) from a generic template.
Builds a temporary wrapper XML that <include>s the real model and adds one
fixed (non-mocap, always-present) rigid pad on the object's far (+Y) face --
an idealized "virtual thumb" -- then runs the SAME retract+rotation test used
throughout this project, for a device that has NO real opposing digit
(D4_v2_hybrid_per_finger_calibrated) with vs without the pad.

How to read it:
  - If the object HOLDS (or rotates far less) WITH the idealized pad but still
    fails without it -> opposition (having ANY contact on the far face) is the
    binding constraint, and the remaining gap in the real extra-thumb device
    is about that contact's QUALITY/timing/coverage, not needing a bigger
    redesign.
  - If it still fails about the same WITH the idealized pad -> even perfect,
    always-on opposition isn't enough, meaning the finger contact geometry
    itself (normals, distribution across height) is the real limiting factor,
    not the presence/absence of a far-side contact.

Does not modify any existing model file -- the wrapper is written to a temp
file next to the real model (so relative asset paths still resolve) and
deleted immediately after loading.
"""

from __future__ import annotations

import os
import tempfile

import mujoco
import numpy as np

from exo_devices import DEVICES, ExoApplicator, JOINT_NAMES
from grasp_controller import GraspController
from grasp_env import MODEL_PATH
from run_grasp_baseline import OBJECT_VARIANTS, configure_object_variant

DEVICE_NAME = "D4_v2_hybrid_per_finger_calibrated"  # no real opposing digit -- known to fail every time
OBJECT_NAME = "tall_box"

# Box center Y=-0.490, half-width 0.036 -> far (+Y) face at world Y=-0.454.
# Pad sits just against that face, matching the object's X/Z footprint with margin.
PAD_POS = np.array([-0.160, -0.449, 1.387])
PAD_SIZE = np.array([0.03, 0.005, 0.05])

RETRACT_OFFSET = np.array([1.0, 0.0, 0.0])
MAX_STEPS = 4000


def build_model(add_pad):
    if not add_pad:
        return mujoco.MjModel.from_xml_path(MODEL_PATH)

    abs_base = os.path.abspath(MODEL_PATH)
    px, py, pz = PAD_POS.tolist()
    sx, sy, sz = PAD_SIZE.tolist()
    # Excludes contact with radius (forearm): the pad's footprint, sized to
    # cover the object, also geometrically overlaps the forearm's collision
    # mesh near the wrist (the same "collision geometry extends well beyond
    # body-origin position" surprise found earlier this project with this
    # same bone) -- without the exclude, the pad collides with the forearm
    # on first contact and perturbs the whole arm's pose for the rest of the
    # run, contaminating the actual question being asked here.
    wrapper = f"""<mujoco>
  <include file="{abs_base}"/>
  <worldbody>
    <body name="opposition_pad" pos="{px} {py} {pz}">
      <geom name="opp_pad_geom" type="box" size="{sx} {sy} {sz}"
            contype="1" conaffinity="1" condim="4"
            friction="2.5 0.02 0.002" rgba="0.9 0.2 0.2 0.5"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="opposition_pad" body2="radius"/>
  </contact>
</mujoco>"""
    out_dir = os.path.dirname(abs_base)
    fd, path = tempfile.mkstemp(suffix=".xml", dir=out_dir, text=True)
    with os.fdopen(fd, "w") as f:
        f.write(wrapper)
    try:
        model = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    return model


def quat_angle_diff(q1, q2):
    dot = min(1.0, abs(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(dot)))


def run_trial(add_pad):
    model = build_model(add_pad)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    variant = next(v for v in OBJECT_VARIANTS if v["name"] == OBJECT_NAME)
    configure_object_variant(model, data, variant)

    obj_geom_id = model.geom("grasp_object_geom").id
    obj_body_id = model.body("grasp_object").id
    device = DEVICES[DEVICE_NAME]
    applicator = ExoApplicator(model)
    controller = GraspController(model, device, obj_geom_id, **device.controller_overrides)

    # DEVICE_NAME doesn't drive extra_thumb (K row all-zero for it), but the
    # joint still physically exists in the shared model and has a small
    # passive droop under gravity even with its spring-return (verified: the
    # spring stiffness that lets a REAL extra-thumb device reach its full 80mm
    # under 15.8N necessarily allows a few mm of undriven droop under
    # gravity). That droop happened to land right at this diagnostic's pad
    # location, producing a spurious one-off collision unrelated to the
    # question being asked here. Pin it to 0 explicitly for this diagnostic
    # only, rather than over-tightening the shared model's spring for an
    # edge case that doesn't come up in any real device test.
    extra_thumb_id = model.joint("extra_thumb").id
    extra_thumb_qposadr = model.joint("extra_thumb").qposadr[0]
    extra_thumb_dofadr = model.joint("extra_thumb").dofadr[0]

    pillar_id = model.body("start_pillar").id
    pillar_mocap_id = model.body("start_pillar").mocapid[0]
    original_pillar_pos = model.body_pos[pillar_id].copy()

    retracted = False
    retract_step = None
    pos_at_retract = None
    quat_at_retract = None
    rot_at = {}
    lin_at = {}
    for step in range(MAX_STEPS):
        u = controller.step(model, data)
        applicator.apply(data, device, u=u, row_gate=controller.row_gate)
        extra_thumb_row = JOINT_NAMES.index("extra_thumb")
        if device.K[extra_thumb_row, :].sum() == 0.0:  # device doesn't drive extra_thumb
            data.qpos[extra_thumb_qposadr] = 0.0
            data.qvel[extra_thumb_dofadr] = 0.0
        mujoco.mj_step(model, data)
        if not retracted and all(controller.locked) and step > 100:
            data.mocap_pos[pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
            retracted = True
            retract_step = step
            pos_at_retract = data.xpos[obj_body_id].copy()
            quat_at_retract = data.xquat[obj_body_id].copy()
        if retracted and (step - retract_step) in (100, 300, 600, 1000, 1500):
            lin_at[step - retract_step] = float(np.linalg.norm(data.xpos[obj_body_id] - pos_at_retract))
            rot_at[step - retract_step] = quat_angle_diff(data.xquat[obj_body_id], quat_at_retract)
        if retracted and step - retract_step >= 1500:
            break
    return retract_step, lin_at, rot_at


def main():
    print(f"Testing {DEVICE_NAME} on {OBJECT_NAME}: no opposition vs idealized fixed pad on the far face.\n")
    for add_pad, label in [(False, "no opposition (baseline)"), (True, "with idealized fixed pad")]:
        retract_step, lin_at, rot_at = run_trial(add_pad)
        print(f"--- {label} ---")
        if retract_step is None:
            print("  never fully locked")
            continue
        for k in sorted(lin_at):
            print(f"  +{k:4d} steps ({k*0.002:.2f}s)  lin_drift={lin_at[k]*100:6.2f}cm  rot={rot_at[k]:6.1f}deg")
        print()


if __name__ == "__main__":
    main()
