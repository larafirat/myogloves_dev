"""Computes real per-joint moment arms from MyoHand's own flexor tendons (FDP2-FDP5
for the fingers, FPL for the thumb), via finite differences on tendon length vs. joint
angle. This is the source of the MOMENT_ARMS_MM constants in exo_devices.py -- rerun
this if the model geometry changes.

FDP (profundus) is used as the anatomical proxy for an external, fingertip-anchored
exo cable: soft/biomimetic exoskeleton designs commonly route their tendons to mimic
the natural FDS/FDP path, so its moment arm is a literature-supported stand-in for a
device's real (unmeasured) moment arm. FPL is the thumb's analogous single flexor,
crossing all four modeled thumb DoFs including cmc_abduction.

Ring/little were added once D1 (Zhao et al. 2025) turned out to be a FIVE-finger
device -- before that, no device in the set drove those digits from real data, and
D4_v2's shared ulnar channel had to use a proxy profile instead.
"""

import mujoco

MODEL_PATH = "myogloves_dev/models/myohand_exoglove.xml"
DIGITS = {
    "FDP2_tendon": ["mcp2_flexion", "pm2_flexion", "md2_flexion"],      # index
    "FDP3_tendon": ["mcp3_flexion", "pm3_flexion", "md3_flexion"],      # middle
    "FDP4_tendon": ["mcp4_flexion", "pm4_flexion", "md4_flexion"],      # ring
    "FDP5_tendon": ["mcp5_flexion", "pm5_flexion", "md5_flexion"],      # little
    "FPL_tendon": ["cmc_abduction", "cmc_flexion", "mp_flexion", "ip_flexion"],  # thumb
}


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

    for tendon, joints in DIGITS.items():
        for joint in joints:
            r = moment_arm_mm(model, data, tendon, joint)
            print(f'    "{joint}": {r:.3f},')
