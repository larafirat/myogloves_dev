"""Validates D3 (Thimabut, Terachinda & Kitisomprayoonkul, Rehabilitation
Research and Practice 2022, Art. 3738219): single input, both fingers coupled,
PIP-peaked coupling from the paper's reported max flexion angles (52 deg MCP,
80 deg PIP, 75 deg DIP) -> K3 = [0.65, 1.00, 0.9375]. Not literally uniform
(that was a placeholder assumption before this paper's numbers were confirmed).
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D3_uniform_single_dof"]
    qpos = run_device(device, u=[1.0], tau_max=0.03)  # kept below saturation, see view_exo_device.py
    print("D3 final qpos:", qpos)

    # PIP (K=1.00) should lead, DIP (0.9375) close behind, MCP (0.65) trails.
    assert qpos["pm2_flexion"] > qpos["mcp2_flexion"], "index PIP should out-flex MCP"
    assert qpos["pm3_flexion"] > qpos["mcp3_flexion"], "middle PIP should out-flex MCP"
    assert qpos["pm2_flexion"] >= qpos["md2_flexion"], "index PIP should not trail DIP"
    assert qpos["pm3_flexion"] >= qpos["md3_flexion"], "middle PIP should not trail DIP"

    print("D3 PASS: PIP-peaked flexion (per Thimabut et al.'s reported ROM) confirmed on both fingers.")
