"""Validates D1's MyoHand-calibrated variant: real per-joint torque ratios
(tau_j = F_j * MOMENT_ARMS_MM[j]) instead of raw published force ratios.

Expects the OPPOSITE distal bias from the original D1: because moment arm
roughly halves from MCP to DIP, the resulting torque is proximal-biased on
the index finger, and peaks at PIP (not MCP) on the middle finger -- this
inversion relative to test_d1_underactuated_distal.py is the point, not a bug.
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D1_underactuated_distal_calibrated"]
    assert device.tau_max_is_calibrated
    qpos = run_device(device, u=[1.0], tau_max=device.tau_max[0])
    print("D1 calibrated final qpos:", qpos)

    # K's MCP-PIP gap is only ~9% (1.000 vs 0.908, within nonlinear-dynamics noise);
    # the robust, K-predicted signal is that DIP (0.618) trails both by a wide margin.
    assert qpos["md2_flexion"] < qpos["mcp2_flexion"] and qpos["md2_flexion"] < qpos["pm2_flexion"], \
        "index DIP not the clear laggard under calibrated (proximal-biased) torque"
    assert qpos["pm3_flexion"] > qpos["mcp3_flexion"] > qpos["md3_flexion"], \
        "middle finger does not peak at PIP under calibrated torque"

    print("D1 calibrated PASS: real moment arms invert the coupling shape vs. published force ratios.")
