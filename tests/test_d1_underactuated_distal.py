"""Validates D1 (Zhao, Guo, Yang, Li, Zhao, Qu, Li, Liu, Wang & Bu, "A novel
underactuated exoskeleton rehabilitation glove for hand flexion and extension
training," Biomimetic Intelligence and Robotics 5, 100248, 2025): single
input, distal-biased coupling.

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
    for name in ("mcp2_flexion", "pm2_flexion", "md2_flexion", "mcp3_flexion", "pm3_flexion", "md3_flexion"):
        assert qpos[name] > 0.5, f"{name} did not flex under positive input"
    for name in ("cmc_flexion", "mp_flexion", "ip_flexion"):
        assert abs(qpos[name]) < 0.1, f"{name} moved but D1 has no thumb channel"
    print("D1 PASS: distal-biased flexion confirmed on both fingers, thumb untouched.")
