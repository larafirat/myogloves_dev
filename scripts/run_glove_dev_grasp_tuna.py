"""Grasp-hold test for myohand_glove_dev_tuna.xml -- the tuna-fish-can
object-variant of run_glove_dev_grasp.py (see that file's module docstring
for the full protocol description; this is a direct adaptation, not a
redesign).

Only real differences from run_glove_dev_grasp.py:
- MODEL_PATH points at the tuna-can variant (myohand_glove_dev_tuna.xml).
- Object name is "007_tuna_fish_can", a SHORT/WIDE cylinder (radius 4.2cm,
  half-height 1.6cm -- 3.2cm tall, 8.4cm diameter), a genuinely different
  shape class from the tall/narrow tomato soup can or the standing box.
- The mass/inertia fix uses the cylinder inertia formulas (I_axial =
  0.5*m*r^2, I_radial = m*(3*r^2+h^2)/12). The tuna can has the SAME
  visual-mesh-mass-inflation bug the box and tomato-soup-can both had:
  compiled body_mass came out to 0.349kg, not the intended 0.171kg (the
  vendored body_007_tuna_fish_can.xml's visual mesh geom has no explicit
  mass, so MuJoCo auto-added its own mesh-volume contribution on top of the
  collision cylinder's explicit mass="0.171", same bug pattern, third
  object in a row). Fixed the same way: override
  body_mass/inertia/ipos/iquat to match ONLY the intended collision
  cylinder.
- No orientation/euler fix needed (the cylinder is already naturally upright
  by default -- see myohand_glove_dev_tuna.xml's own comment on this).

Everything else (arm stabilization, OP pre-shape, hand-joint damping boost,
retract protocol, fail thresholds, HAND_BODY_NAMES/CHANNEL_BODIES) is
identical to run_glove_dev_grasp.py -- this is testing whether the SAME
tuned controller/rig generalizes to a new object shape, not re-deriving a
new controller from scratch.

Usage:
    python run_glove_dev_grasp_tuna.py                 open-loop ramp (default)
    python run_glove_dev_grasp_tuna.py --controller     closed-loop ease-off-on-contact
    python run_glove_dev_grasp_tuna.py --view            add either flag to also
                                                          open the interactive viewer
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "myogloves_dev/models/myohand_glove_dev_tuna.xml"

ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
ARM_AUTHORED_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
ARM_STIFF_BOOST = 8.0
ARM_KV_RATIO = 0.3

OP_CTRL = 0.5

RAMP_SECONDS = 4.0
SETTLE_SECONDS = 1.6
SETTLE_SECONDS_CONTROLLER = 0.15
RETRACT_OFFSET = np.array([0.5, 0.0, 0.0])
POST_RETRACT_SECONDS = 3.0
FAIL_ROT_DEG = 45.0
FAIL_DRIFT_M = 0.05
HOLD_FRACTION = 1.0

HAND_BODY_NAMES = [
    "firstmc", "proximal_thumb", "distal_thumb",
    "secondmc", "proxph2", "midph2", "distph2",
    "thirdmc", "proxph3", "midph3", "distph3",
]

CHANNEL_BODIES = {
    "flex": {"secondmc", "proxph2", "midph2", "distph2", "thirdmc", "proxph3", "midph3", "distph3"},
    "thumb": {"firstmc", "proximal_thumb", "distal_thumb"},
}

OBJECT_NAME = "007_tuna_fish_can"


def quat_angle_deviation(q0, q):
    dot = min(1.0, abs(float(np.dot(q0, q))))
    return np.degrees(2 * np.arccos(dot))


def main():
    interactive = "--view" in sys.argv
    use_controller = "--controller" in sys.argv

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Same class of bug as the gelatin box and tomato-soup can (see
    # run_glove_dev_grasp.py): the tuna can's compiled mass was 0.349kg, not
    # the intended 0.171kg (collision cylinder's stated mass) -- the visual
    # mesh geom has no explicit mass, so its own auto-computed mesh-volume
    # mass got silently added on top. Overriding body_mass/inertia/ipos to
    # match ONLY the intended collision cylinder (a property of our own test
    # prop, not the exoglove).
    obj_body_id_early = model.body(OBJECT_NAME).id
    can_mass = 0.171
    radius, half_height = 0.042, 0.016
    full_height = 2 * half_height
    model.body_mass[obj_body_id_early] = can_mass
    i_axial = 0.5 * can_mass * radius ** 2
    i_radial = can_mass * (3 * radius ** 2 + full_height ** 2) / 12
    model.body_inertia[obj_body_id_early] = [i_radial, i_radial, i_axial]
    model.body_ipos[obj_body_id_early] = [0, 0, half_height]
    model.body_iquat[obj_body_id_early] = [1, 0, 0, 0]

    # Same friction fix as the box test -- a property of our own test prop.
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == obj_body_id_early:
            model.geom_friction[gid] = [2.5, 0.02, 0.002]

    # CONTACT STIFFENING -- see run_glove_dev_grasp_can.py's block for the
    # full reasoning. MyoHand's soft bone contact model (solimp=[0.8,0.8,
    # 0.01]) gets crushed through by EXO_FLEX/EXO_THUMB at full activation;
    # measured penetration into this tuna can reached 32.2mm. Stiffening the
    # contact solver is a contact-model fidelity setting, not an exoglove
    # force/lengthrange/tendon/routing value.
    for gid in range(model.ngeom):
        model.geom_solref[gid] = [0.01, 1.0]
        model.geom_solimp[gid] = [0.98, 0.999, 0.0005, 0.5, 2.0]

    # Same hand/wrist damping boost as run_glove_dev_grasp.py (see its
    # comment for the full story) -- a "soft tissue" modeling assumption, not
    # an exoglove value.
    HAND_DAMPING_BOOST = 25.0
    for jid in range(model.njnt):
        jname = model.joint(jid).name
        if jname in ARM_JOINTS or jname.startswith("OBJ"):
            continue
        dof_start = model.jnt_dofadr[jid]
        dof_count = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
                     mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[model.jnt_type[jid]]
        for d in range(dof_start, dof_start + dof_count):
            model.dof_damping[d] *= HAND_DAMPING_BOOST

    for j in ARM_JOINTS:
        aid = model.actuator(f"A_{j}").id
        kp = ARM_AUTHORED_KP[j] * ARM_STIFF_BOOST
        model.actuator_gainprm[aid][0] = kp
        model.actuator_biasprm[aid][1] = -kp
        model.actuator_biasprm[aid][2] = -kp * ARM_KV_RATIO
        data.ctrl[aid] = 0.0
    op_id = model.actuator("OP").id
    data.ctrl[op_id] = OP_CTRL
    mujoco.mj_forward(model, data)
    for _ in range(6000):
        mujoco.mj_step(model, data)

    exo_flex_id = model.actuator("EXO_FLEX").id
    exo_thumb_id = model.actuator("EXO_THUMB").id

    obj_body_id = model.body(OBJECT_NAME).id
    obj_geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == obj_body_id]

    hand_body_ids = {model.body(n).id for n in HAND_BODY_NAMES}
    channel_body_ids = {
        ch: {model.body(n).id for n in names} for ch, names in CHANNEL_BODIES.items()
    }

    pillar_body = model.body("glove_dev_support_pillar")
    pillar_mocap_id = pillar_body.mocapid[0]
    original_pillar_pos = data.mocap_pos[pillar_mocap_id].copy()

    ramp_step = model.opt.timestep / RAMP_SECONDS
    settle_seconds = SETTLE_SECONDS_CONTROLLER if use_controller else SETTLE_SECONDS
    settle_steps = int(settle_seconds / model.opt.timestep)
    post_retract_steps = int(POST_RETRACT_SECONDS / model.opt.timestep)

    u = {"flex": 0.0, "thumb": 0.0}
    locked = {"flex": False, "thumb": False}
    first_touch_step = None
    settle_start_step = None
    retract_step = None
    pos_at_retract = None
    quat_at_retract = None
    fail_step = None

    max_steps = 20000

    viewer_ctx = None
    if interactive:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)

    step = 0
    while step < max_steps:
        touched = {"flex": False, "thumb": False}
        for i in range(data.ncon):
            c = data.contact[i]
            other = None
            if c.geom1 in obj_geom_ids and model.geom_bodyid[c.geom2] in hand_body_ids:
                other = model.geom_bodyid[c.geom2]
            elif c.geom2 in obj_geom_ids and model.geom_bodyid[c.geom1] in hand_body_ids:
                other = model.geom_bodyid[c.geom1]
            if other is None:
                continue
            for ch, bodies in channel_body_ids.items():
                if other in bodies:
                    touched[ch] = True
        hand_touch = touched["flex"] or touched["thumb"]

        if use_controller:
            for ch in ("flex", "thumb"):
                if touched[ch] and not locked[ch]:
                    locked[ch] = True
                u[ch] = min(u[ch] + ramp_step, 1.0)
        else:
            for ch in ("flex", "thumb"):
                u[ch] = min(u[ch] + ramp_step, 1.0)

        data.ctrl[exo_flex_id] = u["flex"]
        data.ctrl[exo_thumb_id] = u["thumb"]
        mujoco.mj_step(model, data)

        if hand_touch and first_touch_step is None:
            first_touch_step = step

        FULL_FORCE_THRESHOLD = 0.95
        full_grasp_now = (touched["flex"] and touched["thumb"]
                           and u["flex"] >= FULL_FORCE_THRESHOLD
                           and u["thumb"] >= FULL_FORCE_THRESHOLD) if use_controller else hand_touch
        if full_grasp_now and retract_step is None:
            if settle_start_step is None:
                settle_start_step = step
                print(f"{'full grasp' if use_controller else 'first contact'} "
                      f"at t={step * model.opt.timestep:.3f}s; settling...")
            if step - settle_start_step >= settle_steps:
                data.mocap_pos[pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
                retract_step = step
                pos_at_retract = data.xpos[obj_body_id].copy()
                quat_at_retract = data.xquat[obj_body_id].copy()
                print(f"pillar retracted at t={step * model.opt.timestep:.3f}s")
        elif use_controller and not full_grasp_now and retract_step is None:
            if settle_start_step is not None:
                print(f"lost contact at t={step * model.opt.timestep:.3f}s "
                      f"(touched={touched}); resetting settle timer")
            settle_start_step = None

        if retract_step is not None and fail_step is None:
            rot_dev = quat_angle_deviation(quat_at_retract, data.xquat[obj_body_id])
            drift = np.linalg.norm(data.xpos[obj_body_id] - pos_at_retract)
            if rot_dev > FAIL_ROT_DEG or drift > FAIL_DRIFT_M:
                fail_step = step
                print(f"grip failed at t={step * model.opt.timestep:.3f}s "
                      f"(rot_dev={rot_dev:.1f}deg drift={drift:.3f}m)")

        if viewer_ctx is not None:
            viewer_ctx.sync()
            time.sleep(0.002)
            if not viewer_ctx.is_running():
                break

        if retract_step is not None and step - retract_step >= post_retract_steps and viewer_ctx is None:
            break

        step += 1

    if viewer_ctx is not None:
        print("Test finished -- close the viewer window when you're done looking.")
        while viewer_ctx.is_running():
            mujoco.mj_step(model, data)
            viewer_ctx.sync()
            time.sleep(0.002)
        viewer_ctx.close()

    print("=" * 60)
    if first_touch_step is None:
        print("RESULT: never made contact with the object")
        return
    print(f"first contact: t={first_touch_step * model.opt.timestep:.3f}s")
    if retract_step is None:
        print("RESULT: contact made but grip never settled long enough to retract")
        return
    print(f"retracted at: t={retract_step * model.opt.timestep:.3f}s")
    if fail_step is None:
        hold_s = (step - retract_step) * model.opt.timestep
        print(f"RESULT: held for the full {hold_s:.3f}s post-retract window (never failed)")
    else:
        hold_s = (fail_step - retract_step) * model.opt.timestep
        print(f"RESULT: held for {hold_s:.3f}s before drift/rotation exceeded threshold")


if __name__ == "__main__":
    main()
