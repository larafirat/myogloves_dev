"""Live viewer for the grasp scene (models/myohand_exoglove_grasp.xml): drives
one device's digits closed against the grasp object using the same closed-loop
GraspController grasp_test.py uses (ramp -> ease off on contact -> grip harder
if slip is detected), so what you see here matches the numbers, not the old
open-loop "ramp to max and hold" behavior that let objects slip out.

Usage (must run with mjpython on macOS for the interactive window):
    mjpython myogloves_dev/scripts/view_grasp_test.py [DEVICE_NAME]

Defaults to D1_underactuated_distal if no device name is given. Only D4's two
variants have n_inputs=3 (index/middle/thumb, real thumb per Gerez et al.
2020); D1-D3 have n_inputs=1 (index+middle together, no thumb).
"""

import sys
import time

import mujoco
import mujoco.viewer

from exo_devices import DEVICES, ExoApplicator
from grasp_controller import GraspController

MODEL_PATH = "myogloves_dev/models/myohand_exoglove_grasp.xml"

device_name = sys.argv[1] if len(sys.argv) > 1 else "D1_underactuated_distal"
device = DEVICES[device_name]
if not device.tau_max_is_calibrated:
    device.tau_max[:] = 0.03  # kept below joint-limit saturation, same convention as view_exo_device.py

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
applicator = ExoApplicator(model)
obj_geom_id = model.geom("grasp_object_geom").id
controller = GraspController(model, device, obj_geom_id)

print(f"Viewing {device_name} (tau_max={list(device.tau_max)}) with closed-loop grip control.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        u = controller.step(model, data)
        applicator.apply(data, device, u=u, row_gate=controller.row_gate)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
