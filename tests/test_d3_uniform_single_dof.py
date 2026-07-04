"""Validates D3 (Thimabut et al. 2022): single input, uniform coupling.
K3 = [1, 1, 1] per finger -- no distal bias, all three joints track together.
"""

from common import run_device
from exo_devices import DEVICES

TOLERANCE = 0.15  # rad; a looser bound below joint-limit saturation, where equal K still
                  # leaves small qpos differences from each joint's own damping/inertia

if __name__ == "__main__":
    device = DEVICES["D3_uniform_single_dof"]
    qpos = run_device(device, u=[1.0], tau_max=0.03)  # kept below saturation, see view_exo_device.py
    print("D3 final qpos:", qpos)

    index_vals = [qpos["mcp2_flexion"], qpos["pm2_flexion"], qpos["md2_flexion"]]
    middle_vals = [qpos["mcp3_flexion"], qpos["pm3_flexion"], qpos["md3_flexion"]]

    assert max(index_vals) - min(index_vals) < TOLERANCE, "index joints not uniform"
    assert max(middle_vals) - min(middle_vals) < TOLERANCE, "middle joints not uniform"

    print("D3 PASS: uniform flexion confirmed across MCP/PIP/DIP on both fingers.")
