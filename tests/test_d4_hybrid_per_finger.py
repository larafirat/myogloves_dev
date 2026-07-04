"""Validates D4 (NewDexterity Hybrid): two independent inputs, block-diagonal K,
distal-biased within each finger, zero cross-finger coupling.
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D4_hybrid_per_finger"]

    index_only = run_device(device, u=[1.0, 0.0], tau_max=0.03)
    print("D4 index-only drive qpos:", index_only)
    assert index_only["mcp2_flexion"] > 0.5, "index did not flex when driven"
    assert index_only["mcp2_flexion"] < index_only["pm2_flexion"] < index_only["md2_flexion"], \
        "index finger not distal-biased"
    assert index_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"
    assert index_only["pm3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"
    assert index_only["md3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"

    middle_only = run_device(device, u=[0.0, 1.0], tau_max=0.03)
    print("D4 middle-only drive qpos:", middle_only)
    assert middle_only["mcp3_flexion"] > 0.5, "middle did not flex when driven"
    assert middle_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert middle_only["pm2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert middle_only["md2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"

    print("D4 PASS: independent per-finger control confirmed, zero cross-coupling leak.")
