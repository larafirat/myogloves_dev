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
    """SIGNED moment arm, mm/rad. The sign is the whole point and must not be
    discarded (this function used to return abs(), which was a real bug -- see
    below).

    dL/dq is how much the tendon LENGTHENS as the joint coordinate increases.
    A tendon can only pull, so tension always acts to shorten it, and the
    torque it produces therefore has the OPPOSITE sign to dL/dq. Returning
    -dL/dq gives a number that is both the moment arm's magnitude and the
    direction the tendon actually drives the joint.

    Why this matters: MyoHand's joint coordinates do not share a consistent
    "positive = flexion" convention. For all twelve finger joints, positive q
    is flexion. For the thumb it is mixed -- cmc_abduction, mp_flexion and
    ip_flexion all flex NEGATIVE, while cmc_flexion flexes positive. Taking
    abs() here silently made every thumb row positive in exo_devices.py, so
    every device that drives the thumb was driving it backwards: measured
    against the healthy hand's own flexors, the all-positive thumb drive moved
    the thumb tip 45mm AWAY from the index fingertip and 42mm away from the
    middle, where the sign-correct drive moves it 33mm and 51mm TOWARD them.
    Devices were extending and hyperabducting the thumb during "grasp".

    Verified against MyoHand's own muscles: the signs below reproduce exactly
    what FPL does to the thumb (cmc_abd -38.6, cmc_flex +24.9, mp -62.4,
    ip -85.4 deg) and what FDP2 does to the index (all +90 deg).
    """
    mujoco.mj_resetData(model, data)
    tid = model.tendon(tendon_name).id
    qadr = model.joint(joint_name).qposadr[0]
    mujoco.mj_forward(model, data)
    length_0 = data.ten_length[tid]
    data.qpos[qadr] += dq
    mujoco.mj_forward(model, data)
    length_1 = data.ten_length[tid]
    return -(length_1 - length_0) / dq * 1000.0  # m/rad -> mm/rad, signed


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    for tendon, joints in DIGITS.items():
        for joint in joints:
            r = moment_arm_mm(model, data, tendon, joint)
            print(f'    "{joint}": {r:.3f},')
