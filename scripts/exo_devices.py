"""Applies exoskeleton coupling-matrix torques (tau_exo = K @ u) to MyoHand.

Joint order (index + middle, 6 DoF):
    index MCP, index PIP, index DIP, middle MCP, middle PIP, middle DIP
Positive torque = flexion assistance.

tau_max (N*m) is a calibration input, not a published number: the source
papers report tendon *forces*, and converting to joint torque requires each
device's real moment arm at the MyoHand joint geometry. Until that's
measured, treat tau_max as a placeholder to tune against observed grasp
behavior, and revisit per the moment-arm caveat before citing results.
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


class ExoDevice:
    def __init__(self, name, K, tau_max, n_inputs):
        self.name = name
        self.n_inputs = n_inputs
        self.K = np.asarray(K, dtype=float).reshape(len(JOINT_NAMES), n_inputs)
        self.tau_max = np.asarray(tau_max, dtype=float).reshape(n_inputs)

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
