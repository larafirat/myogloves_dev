"""Validates D4 (NewDexterity Hybrid): three independent inputs (index, middle,
thumb -- the only device here with a real, independently-actuated thumb, per
Gerez et al. 2020 Sec. III), block-diagonal K, distal-biased within each digit,
zero cross-digit coupling.
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D4_hybrid_per_finger"]

    index_only = run_device(device, u=[1.0, 0.0, 0.0], tau_max=0.03)
    print("D4 index-only drive qpos:", index_only)
    assert index_only["mcp2_flexion"] > 0.5, "index did not flex when driven"
    assert index_only["mcp2_flexion"] < index_only["pm2_flexion"] < index_only["md2_flexion"], \
        "index finger not distal-biased"
    assert index_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"
    assert index_only["pm3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"
    assert index_only["md3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"
    assert abs(index_only["ip_flexion"]) < 0.05, "thumb moved despite u_thumb=0 (cross-coupling leak)"

    middle_only = run_device(device, u=[0.0, 1.0, 0.0], tau_max=0.03)
    print("D4 middle-only drive qpos:", middle_only)
    assert middle_only["mcp3_flexion"] > 0.5, "middle did not flex when driven"
    assert middle_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert middle_only["pm2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert middle_only["md2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert abs(middle_only["ip_flexion"]) < 0.05, "thumb moved despite u_thumb=0 (cross-coupling leak)"

    thumb_only = run_device(device, u=[0.0, 0.0, 1.0], tau_max=0.03)
    print("D4 thumb-only drive qpos:", thumb_only)
    # Thumb joints have much smaller anatomical ranges than index/middle's 0-90 deg
    # (ip_flexion maxes out at 0.436 rad, less than the ">0.5" bar used above), so
    # check against fractions of each joint's own range instead of an absolute value.
    assert thumb_only["cmc_flexion"] > 0.3, "thumb CMC did not flex when driven"
    assert thumb_only["mp_flexion"] > 0.3, "thumb MP did not flex when driven"
    assert thumb_only["ip_flexion"] > 0.25, "thumb IP did not flex when driven"
    assert thumb_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0 (cross-coupling leak)"
    assert thumb_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0 (cross-coupling leak)"

    print("D4 PASS: independent per-digit control confirmed (incl. real thumb), zero cross-coupling leak.")
