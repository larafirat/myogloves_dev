"""Simple static viewer for myohand_glove_dev.xml -- shows the current
stabilized-arm + OP-preshaped setup WITHOUT running the retract-and-hold
grasp test (no pillar retraction, no drift/rotation measurement). Useful for
just looking at the pose/setup; use run_glove_dev_grasp.py --view for the
actual measured grasp test.

Usage:
    mjpython view_glove_dev_static.py
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

MODEL_PATH = "myogloves_dev/models/myohand_glove_dev.xml"
ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
ARM_AUTHORED_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
ARM_STIFF_BOOST = 8.0
ARM_KV_RATIO = 0.3
OP_CTRL = 0.5


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Same mass-bug fix as run_glove_dev_grasp.py (see its comment): the
    # object's visual mesh geom has no explicit mass, so MuJoCo silently
    # added its own mesh-volume mass on top of the collision geom's stated
    # 0.097kg, inflating the compiled mass to 0.27kg.
    obj_body_id = model.body("009_gelatin_box").id
    box_mass = 0.097
    hx, hy, hz = 0.036, 0.044, 0.014
    model.body_mass[obj_body_id] = box_mass
    model.body_inertia[obj_body_id] = [
        box_mass * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12,
        box_mass * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12,
        box_mass * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12,
    ]
    model.body_ipos[obj_body_id] = [0, 0, hz]
    model.body_iquat[obj_body_id] = [1, 0, 0, 0]
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == obj_body_id:
            model.geom_friction[gid] = [2.5, 0.02, 0.002]

    # Same contact stiffening as run_glove_dev_grasp.py (see its comment for
    # the full story): MyoHand's soft bone contact model gets crushed through
    # by the exoglove's ~200 N, letting fingers penetrate the object by up to
    # 16.8mm -- the "transparent hand" artifact. Kept in sync here so the
    # static viewer shows the same physics as the measured test.
    for gid in range(model.ngeom):
        model.geom_solref[gid] = [0.01, 1.0]
        model.geom_solimp[gid] = [0.98, 0.999, 0.0005, 0.5, 2.0]

    # Extra passive damping on the hand/wrist joints (not the arm mount,
    # handled separately below) -- see run_glove_dev_grasp.py's comment for
    # the full story: OP is a muscle with its own activation dynamics, so the
    # thumb builds up real velocity approaching the object and can overshoot
    # into it before contact forces stop it, producing a violent startup
    # kick. More damping doesn't prevent that initial contact-driven
    # transient but drains it out much faster and more completely
    # afterward (residual velocity: 0.08 at 8x boost vs 0.003 at 25x,
    # stable from ~step 3000 onward). A physically reasonable "soft tissue"
    # assumption, not an exoglove value.
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

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for j in ARM_JOINTS:
        aid = model.actuator(f"A_{j}").id
        kp = ARM_AUTHORED_KP[j] * ARM_STIFF_BOOST
        model.actuator_gainprm[aid][0] = kp
        model.actuator_biasprm[aid][1] = -kp
        model.actuator_biasprm[aid][2] = -kp * ARM_KV_RATIO
        data.ctrl[aid] = 0.0
    data.ctrl[model.actuator("OP").id] = OP_CTRL
    mujoco.mj_forward(model, data)

    # Pre-settle headlessly before opening the viewer: the object (resting
    # against the OP-preshaped thumb) has a real settling transient (angular
    # velocity spikes around step 1000-2500) that needs the damping boost
    # above plus this many steps to fully damp out. Opening the viewer any
    # earlier shows this ongoing motion as a startup "jump".
    for _ in range(6000):
        mujoco.mj_step(model, data)

    print("=" * 60)
    print("STATIC VIEW -- stabilized arm + OP thumb pre-shape, no grasp test")
    print("(no pillar retraction, no exo ramp -- just the settled pose)")
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
            step += 1


if __name__ == "__main__":
    main()
