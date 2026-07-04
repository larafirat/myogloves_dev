"""Applies exoskeleton coupling-matrix torques (tau_exo = K @ u) to MyoHand.

Joint order (index + middle, 6 DoF):
    index MCP, index PIP, index DIP, middle MCP, middle PIP, middle DIP
Positive torque = flexion assistance.

tau_max (N*m) is mostly still a placeholder tuned for visible grasp behavior in
the demo/tests. The two "_calibrated" device variants below are the exception:
their K and tau_max are derived from real MyoHand moment arms (MOMENT_ARMS_MM),
not tuned by hand -- see the comment block above them for how and why.
"""

import numpy as np

JOINT_NAMES = [
    "mcp2_flexion",  # index MCP
    "pm2_flexion",   # index PIP
    "md2_flexion",   # index DIP
    "mcp3_flexion",  # middle MCP
    "pm3_flexion",   # middle PIP
    "md3_flexion",   # middle DIP
]

# Real per-joint moment arms (mm), from MyoHand's own FDP2/FDP3 tendons via
# compute_moment_arms.py. Used only for the "_calibrated" device variants below;
# these show the moment arm roughly halves from MCP to DIP, so it is NOT constant
# across a finger's joints -- the "moment-arm-invariant" assumption behind the
# literature K matrices does not hold for this anatomy.
MOMENT_ARMS_MM = {
    "mcp2_flexion": 9.583,
    "pm2_flexion": 6.862,
    "md2_flexion": 4.333,
    "mcp3_flexion": 8.457,
    "pm3_flexion": 7.546,
    "md3_flexion": 2.664,
}


class ExoDevice:
    def __init__(self, name, K, tau_max, n_inputs, tau_max_is_calibrated=False):
        self.name = name
        self.n_inputs = n_inputs
        self.K = np.asarray(K, dtype=float).reshape(len(JOINT_NAMES), n_inputs)
        self.tau_max = np.asarray(tau_max, dtype=float).reshape(n_inputs)
        # True only when tau_max is derived from real force/moment-arm data (see D1's
        # calibrated variant) rather than tuned by hand for visible demo behavior.
        self.tau_max_is_calibrated = tau_max_is_calibrated

    def torque(self, u):
        """u: array-like of length n_inputs, each in [0, 1]. Returns (6,) joint torques."""
        u = np.clip(np.asarray(u, dtype=float).reshape(self.n_inputs), 0.0, 1.0)
        return self.K @ (u * self.tau_max)


DEVICES = {
    "D1_underactuated_distal": ExoDevice(
        name="D1_underactuated_distal",
        K=[0.73, 0.93, 1.00, 0.73, 0.93, 1.00],
        tau_max=[1.0],  # placeholder, calibrate against DIP force 5.6 N * real moment arm
        n_inputs=1,
    ),
    "D2_synergy_cross_finger": ExoDevice(
        name="D2_synergy_cross_finger",
        K=[0.85, 0.95, 1.00, 0.85, 0.95, 1.00],
        tau_max=[1.0],  # placeholder; D2 has no published per-joint force split (design-intent parameterized)
        n_inputs=1,
    ),
    "D3_uniform_single_dof": ExoDevice(
        name="D3_uniform_single_dof",
        K=[1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
        tau_max=[1.0],  # placeholder; no published force split, uniform coupling is the design-intent assumption
        n_inputs=1,
    ),
    "D4_hybrid_per_finger": ExoDevice(
        name="D4_hybrid_per_finger",
        K=[
            [0.80, 0.00],
            [0.90, 0.00],
            [1.00, 0.00],
            [0.00, 0.80],
            [0.00, 0.90],
            [0.00, 1.00],
        ],
        tau_max=[1.0, 1.0],  # placeholder; real device has a 500 N tendon rating, highest of the four
        n_inputs=2,
    ),
    # --- MyoHand-calibrated variants ---
    # These replace the literature force ratios with real joint-torque ratios
    # (tau_j = F_j * MOMENT_ARMS_MM[j]), for the two devices with published
    # per-joint/per-tendon force data (D1, D4). D2/D3 have no force data to
    # correct (design-intent only), so no calibrated variant exists for them.
    # Report both the original and calibrated K in the paper: the discrepancy
    # (D1 flips from distal-biased to proximal-biased) is itself a finding.
    "D1_underactuated_distal_calibrated": ExoDevice(
        name="D1_underactuated_distal_calibrated",
        K=[1.000, 0.908, 0.618, 0.883, 0.999, 0.380],
        tau_max=[0.03929],  # N*m; derived from published forces * real MyoHand moment arms, not a placeholder
        n_inputs=1,
        tau_max_is_calibrated=True,
    ),
    "D4_hybrid_per_finger_calibrated": ExoDevice(
        name="D4_hybrid_per_finger_calibrated",
        # Single fingertip tendon under one tension per finger -> K shape here IS
        # the moment-arm profile itself (tau_j = F_max * r_j, same F_max per finger).
        K=[
            [1.000, 0.000],
            [0.716, 0.000],
            [0.452, 0.000],
            [0.000, 1.000],
            [0.000, 0.892],
            [0.000, 0.315],
        ],
        tau_max=[1.0, 1.0],  # still placeholder: the 500 N rating is a tendon material/safety limit,
        n_inputs=2,           # not a typical operating force -- using it directly would be unrealistic
    ),
}


class ExoApplicator:
    """Injects a device's joint torques into MjData via qfrc_applied each step."""

    def __init__(self, model):
        self.dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]

    def apply(self, data, device, u):
        tau = device.torque(u)
        for dof, t in zip(self.dof_adr, tau):
            data.qfrc_applied[dof] = t

    def clear(self, data):
        for dof in self.dof_adr:
            data.qfrc_applied[dof] = 0.0
