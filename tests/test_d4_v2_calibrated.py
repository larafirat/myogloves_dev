"""Validates D4_v2's paper-faithful shared ulnar channel: index, middle, and
thumb remain individually driven, while ring+pinky share one active input.

This keeps the current D4 model untouched and adds a separate version closer
to Gerez et al. 2020 Sec. III, where "the ring and the pinky finger tendons
are coupled together and connected to a single pulley and motor."
"""

from common import run_device
from exo_devices import DEVICES


if __name__ == "__main__":
    device = DEVICES["D4_v2_hybrid_per_finger_calibrated"]
    assert device.tau_max_is_calibrated

    ulnar_only = run_device(device, u=[0.0, 0.0, 0.0, 1.0], tau_max=device.tau_max)
    print("D4_v2 calibrated ulnar-only qpos:", ulnar_only)

    for j in ["mcp4_flexion", "pm4_flexion", "mcp5_flexion", "pm5_flexion"]:
        assert ulnar_only[j] > 0.3, f"{j} did not flex under shared ring+pinky drive"

    assert abs(ulnar_only["mcp2_flexion"]) < 0.05, "index moved despite u_index=0"
    assert abs(ulnar_only["mcp3_flexion"]) < 0.05, "middle moved despite u_middle=0"
    assert abs(ulnar_only["ip_flexion"]) < 0.05, "thumb moved despite u_thumb=0"

    print("D4_v2 PASS: shared active ring+pinky channel works without leaking into index/middle/thumb.")
