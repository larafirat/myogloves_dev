"""Validates D4's MyoHand-calibrated variant: a single fingertip tendon under one
tension per finger means the coupling shape IS the real moment-arm profile
(tau_j = F_max * MOMENT_ARMS_MM[j]), still proximal-biased like D1's calibrated
variant, but independent per finger (two inputs) like the original D4.

tau_max here is still a placeholder -- the device's published 500 N rating is a
tendon material/safety limit, not a realistic operating force, so it is not used
directly (see exo_devices.py comment).
"""

from common import run_device
from exo_devices import DEVICES

if __name__ == "__main__":
    device = DEVICES["D4_hybrid_per_finger_calibrated"]
    assert not device.tau_max_is_calibrated  # confirms magnitude is still a placeholder, shape is not

    index_only = run_device(device, u=[1.0, 0.0], tau_max=0.03)
    print("D4 calibrated index-only qpos:", index_only)
    assert index_only["mcp2_flexion"] > index_only["pm2_flexion"] > index_only["md2_flexion"], \
        "index finger not proximal-biased under calibrated shape"
    assert index_only["mcp3_flexion"] < 0.05, "middle finger moved despite u_middle=0"

    middle_only = run_device(device, u=[0.0, 1.0], tau_max=0.03)
    print("D4 calibrated middle-only qpos:", middle_only)
    assert middle_only["mcp3_flexion"] > middle_only["pm3_flexion"] > middle_only["md3_flexion"], \
        "middle finger not proximal-biased under calibrated shape"
    assert middle_only["mcp2_flexion"] < 0.05, "index finger moved despite u_index=0"

    print("D4 calibrated PASS: proximal-biased, independent per-finger control confirmed.")
