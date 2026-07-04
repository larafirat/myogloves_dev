"""Validates D1 (Sheng et al. 2025): single input, distal-biased coupling.

Expects: positive u drives flexion on both fingers, with settle order
MCP < PIP < DIP on each finger, matching K1 = [0.73, 0.93, 1.00].
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D1_underactuated_distal"]
    qpos = run_device(device, u=[1.0], tau_max=0.03)  # kept below saturation, see view_exo_device.py
    print("D1 final qpos:", qpos)

    assert qpos["mcp2_flexion"] < qpos["pm2_flexion"] < qpos["md2_flexion"], "index finger not distal-biased"
    assert qpos["mcp3_flexion"] < qpos["pm3_flexion"] < qpos["md3_flexion"], "middle finger not distal-biased"
    for name, val in qpos.items():
        assert val > 0.5, f"{name} did not flex under positive input"
    print("D1 PASS: distal-biased flexion confirmed on both fingers.")
