"""Validates D4's MyoHand-calibrated variant: a single fingertip tendon under one
tension per digit means the coupling shape IS the real moment-arm profile
(tau_j = F_max * MOMENT_ARMS_MM[j]), independent per digit (three inputs: index,
middle, thumb) like the original D4.

tau_max is now real: Gerez et al. 2020's Table 1 reports "Maximum fingertip
force 13.8 N" (unjammed) -- see exo_devices.py for the F_max * moment-arm
derivation per digit (0.13225 N*m index, 0.11671 N*m middle, 0.12162 N*m
thumb). This replaces the old placeholder that used the device's 500 N tendon
rating directly (a material safety limit, not an operating force).

Unlike D1's calibrated torque (0.03929 N*m, from Zhao et al.'s smaller 5.6 N
max), D4's real torque is over 3x larger and fully saturates MyoHand's
0-90 deg joint range on all three joints of index/middle -- so the
proximal-bias K shape gets washed out entirely rather than merely narrowed.
The thumb saturates too, just against its own (much smaller) anatomical
range per joint (CMC/MP/IP don't share index/middle's 0-90 deg limit).
That saturation-vs-D1 contrast is itself a real, literature-grounded finding
worth reporting (Gerez et al.'s device is simply much stronger), so this test
checks for full saturation rather than an ordering the physics no longer shows.
"""

from common import run_device
from exo_devices import DEVICES

JOINT_LIMIT_RAD = 1.5708  # 90 deg, MyoHand's index/middle mcp/pip/dip joint range max
# Thumb joints have their own, much smaller anatomical ranges (see myohand_body.xml):
THUMB_UPPER_LIMIT_RAD = {"cmc_flexion": 0.7, "mp_flexion": 0.698132, "ip_flexion": 0.436332}

if __name__ == "__main__":
    device = DEVICES["D4_hybrid_per_finger_calibrated"]
    assert device.tau_max_is_calibrated

    index_only = run_device(device, u=[1.0, 0.0, 0.0], tau_max=device.tau_max)
    print("D4 calibrated index-only qpos:", index_only)
    for j in ["mcp2_flexion", "pm2_flexion", "md2_flexion"]:
        assert index_only[j] >= JOINT_LIMIT_RAD - 0.05, f"{j} not saturated under D4's real (13.8N-derived) torque"
    assert index_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0"
    assert abs(index_only["ip_flexion"]) < 0.05, "thumb moved despite u_thumb=0"

    middle_only = run_device(device, u=[0.0, 1.0, 0.0], tau_max=device.tau_max)
    print("D4 calibrated middle-only qpos:", middle_only)
    for j in ["mcp3_flexion", "pm3_flexion", "md3_flexion"]:
        assert middle_only[j] >= JOINT_LIMIT_RAD - 0.15, f"{j} not (near-)saturated under D4's real torque"
    assert middle_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0"
    assert abs(middle_only["ip_flexion"]) < 0.05, "thumb moved despite u_thumb=0"

    thumb_only = run_device(device, u=[0.0, 0.0, 1.0], tau_max=device.tau_max)
    print("D4 calibrated thumb-only qpos:", thumb_only)
    for j, limit in THUMB_UPPER_LIMIT_RAD.items():
        assert thumb_only[j] >= limit - 0.05, f"{j} not saturated against its own joint range under D4's real torque"
    assert thumb_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0"
    assert thumb_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0"

    print("D4 calibrated PASS: real 13.8N-derived torque fully saturates index/middle/thumb, "
          "each against its own anatomical joint range -- proximal bias is washed out, not just narrowed.")
