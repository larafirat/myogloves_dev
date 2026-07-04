"""Shared helper for headless per-device validation scripts in this directory."""

import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from exo_devices import ExoApplicator, JOINT_NAMES  # noqa: E402

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "myohand_exoglove.xml")


def run_device(device, u, tau_max, steps=1000):
    """Loads a fresh MyoHand, drives `device` with constant input `u` for `steps`,
    and returns the final qpos of the 6 tracked joints, in JOINT_NAMES order."""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    applicator = ExoApplicator(model)
    device.tau_max[:] = tau_max

    for _ in range(steps):
        applicator.apply(data, device, u=u)
        mujoco.mj_step(model, data)

    qadr = [model.joint(name).qposadr[0] for name in JOINT_NAMES]
    return dict(zip(JOINT_NAMES, (round(data.qpos[a], 4) for a in qadr)))
