"""Compare the myohand_glove_dev.xml arm mount WITH vs WITHOUT gravity sag.

The 6-DoF position-servo arm mount (ARTx/y/z, ARRx/y/z) hangs a fairly heavy
forearm+hand assembly off a cantilevered mount. With the position actuators'
authored gains (kp=175 translation / 150 rotation) and ctrl held at the
authored zero setpoint, gravity still wins: it settles at roughly
ARTz=-0.100 (pinned at that joint's own -0.1..0.1 range limit, not a gain
problem), ARRy=-0.27rad (~-15deg droop). That's the "WITH sag" case below --
it's what every earlier grasp test in this project implicitly used, holding
the arm at that settled equilibrium rather than a deliberately-chosen pose.

"WITHOUT sag": boost the same 6 actuators' gain 8x (kp*8) so they can
actually counteract gravity's torque at ctrl=0 -- reduces ARRy droop to
about -0.09rad (~-5deg, a 3x reduction). ARTz still pins at -0.1 regardless
of gain, since that's a hard joint range limit on the rig itself, not
something more actuator strength changes.

Also adds velocity damping (kv = kp*0.3) to those same 6 actuators: boosting
kp alone with NO kv (the authored gains have kv=0, relying only on the
joints' own damping="20") made the system underdamped -- the whole arm
visibly rang/swayed left-right for several seconds after spawning at its
non-equilibrium authored qpos=0 before settling, since the stiffer spring
had nothing extra to dissipate the transient energy. With kv added it
overshoots once, smoothly, and is flat within ~1s (verified: 1 velocity
sign-change in the first second, vs visible sustained oscillation without
it). Model/tendon content is untouched -- this only mutates the arm rig's
own actuator gains at runtime.

Usage:
    python view_glove_dev_arm_pose.py sag       (default arm behavior)
    python view_glove_dev_arm_pose.py nosag      (stiffened, gravity-compensated)
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer

MODEL_PATH = "myogloves_dev/models/myohand_glove_dev.xml"
ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
AUTHORED_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
STIFF_BOOST = 8.0
KV_RATIO = 0.3  # kv = kp * KV_RATIO, only applied in "nosag" mode


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sag"
    if mode not in ("sag", "nosag"):
        raise SystemExit("usage: view_glove_dev_arm_pose.py [sag|nosag]")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for j in ARM_JOINTS:
        aid = model.actuator(f"A_{j}").id
        kp = AUTHORED_KP[j] * (STIFF_BOOST if mode == "nosag" else 1.0)
        model.actuator_gainprm[aid][0] = kp
        model.actuator_biasprm[aid][1] = -kp
        if mode == "nosag":
            model.actuator_biasprm[aid][2] = -kp * KV_RATIO
        data.ctrl[aid] = 0.0

    label = "WITHOUT gravity sag (stiffened, 8x gain)" if mode == "nosag" else "WITH gravity sag (authored gains)"
    print("=" * 60)
    print(label)
    print("=" * 60)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)
            step += 1
            if step % 1000 == 0:
                vals = {j: round(float(data.qpos[model.jnt_qposadr[model.joint(j).id]]), 3) for j in ARM_JOINTS}
                print(f"t={step * model.opt.timestep:.1f}s arm qpos: {vals}")


if __name__ == "__main__":
    main()
