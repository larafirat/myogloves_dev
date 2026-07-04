"""Validates D2's MyoHand-calibrated variant.

D2 has no published per-joint force data, only a design description: a single
continuous tendon anchored at the fingertip, one motor/tension driving both
fingers. Under uniform tension, real joint torque ratio = moment-arm ratio, so
this reuses MOMENT_ARMS_MM directly (same physical assumption as D4's per-finger
calibrated variant, just wired to one shared input instead of two independent
ones). Expects proximal-biased flexion (MCP > PIP > DIP), the opposite of the
original D2's flat/distal-biased design-intent K.
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D2_synergy_cross_finger_calibrated"]
    qpos = run_device(device, u=[1.0], tau_max=0.03)
    print("D2 calibrated final qpos:", qpos)

    assert qpos["mcp2_flexion"] > qpos["pm2_flexion"] > qpos["md2_flexion"], \
        "index finger not proximal-biased under calibrated (real moment-arm) torque"
    assert qpos["mcp3_flexion"] > qpos["pm3_flexion"] > qpos["md3_flexion"], \
        "middle finger not proximal-biased under calibrated (real moment-arm) torque"

    print("D2 calibrated PASS: real moment arms invert the design-intent flat/distal bias to proximal-biased.")
