"""Computes real per-joint moment arms from MyoHand's own flexor tendons (FDP2/FDP3),
via finite differences on tendon length vs. joint angle. This is the source of the
MOMENT_ARMS_MM constants in exo_devices.py -- rerun this if the model geometry changes.

FDP (profundus) is used as the anatomical proxy for an external, fingertip-anchored
exo cable: soft/biomimetic exoskeleton designs commonly route their tendons to mimic
the natural FDS/FDP path, so its moment arm is a literature-supported stand-in for a
device's real (unmeasured) moment arm.
"""

import mujoco

MODEL_PATH = "myogloves_dev/models/myohand_exoglove.xml"
JOINTS = ["mcp2_flexion", "pm2_flexion", "md2_flexion", "mcp3_flexion", "pm3_flexion", "md3_flexion"]
TENDONS = {"2": "FDP2_tendon", "3": "FDP3_tendon"}  # "2" = index, "3" = middle


def moment_arm_mm(model, data, tendon_name, joint_name, dq=1e-4):
    mujoco.mj_resetData(model, data)
    tid = model.tendon(tendon_name).id
    qadr = model.joint(joint_name).qposadr[0]
    mujoco.mj_forward(model, data)
    length_0 = data.ten_length[tid]
    data.qpos[qadr] += dq
    mujoco.mj_forward(model, data)
    length_1 = data.ten_length[tid]
    return abs((length_1 - length_0) / dq) * 1000.0  # m/rad -> mm/rad


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    for joint in JOINTS:
        digit = "2" if "2" in joint else "3"
        r = moment_arm_mm(model, data, TENDONS[digit], joint)
        print(f"{joint}: {r:.3f} mm")
