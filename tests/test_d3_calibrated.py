"""Validates D3's MyoHand-calibrated variant.

D3 (Thimabut et al. 2022) reports real per-joint ROM (52/80/75 deg at
MCP/PIP/DIP -- see D3_uniform_single_dof's comment in exo_devices.py) but no
per-joint FORCE data, and "a single DOF... controls MCP, PIP, DIP" via one
motor. Under the same uniform-tension assumption used for D2/D4, real torque
ratio = moment-arm ratio, so D3's calibrated K is IDENTICAL to D2's -- this
pipeline has no force data to physically distinguish D2's design-intent K
from D3's, which is itself worth reporting rather than papering over.
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    d2 = DEVICES["D2_synergy_cross_finger_calibrated"]
    d3 = DEVICES["D3_uniform_single_dof_calibrated"]
    assert (d2.K == d3.K).all(), "D2 and D3 calibrated K unexpectedly diverged"

    qpos = run_device(d3, u=[1.0], tau_max=0.03)
    print("D3 calibrated final qpos:", qpos)

    assert qpos["mcp2_flexion"] > qpos["pm2_flexion"] > qpos["md2_flexion"], \
        "index finger not proximal-biased under calibrated (real moment-arm) torque"
    assert qpos["mcp3_flexion"] > qpos["pm3_flexion"] > qpos["md3_flexion"], \
        "middle finger not proximal-biased under calibrated (real moment-arm) torque"

    print("D3 calibrated PASS: matches D2 calibrated exactly -- no force data to tell them apart physically.")
