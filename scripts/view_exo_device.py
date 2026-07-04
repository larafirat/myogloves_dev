import sys
import time

import mujoco
import mujoco.viewer

from exo_devices import DEVICES, ExoApplicator

MODEL_PATH = "myogloves_dev/models/myohand_exoglove.xml"
RAMP_SECONDS = 1.5

if len(sys.argv) > 1:
    device_name = sys.argv[1]
else:
    device_name = "D1_underactuated_distal"

device = DEVICES[device_name]
if not device.tau_max_is_calibrated:
    device.tau_max[:] = 0.03  # kept below joint-limit saturation so coupling-shape differences stay visible
# devices with tau_max_is_calibrated=True keep their MyoHand-derived tau_max as-is

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
applicator = ExoApplicator(model)

t = 0.0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if device.n_inputs == 2:
            # stagger the two inputs to show index/middle move independently
            u_index = min(t / RAMP_SECONDS, 1.0)
            u_middle = min(max(t - RAMP_SECONDS, 0.0) / RAMP_SECONDS, 1.0)
            u = [u_index, u_middle]
        else:
            u = [min(t / RAMP_SECONDS, 1.0)] * device.n_inputs
        applicator.apply(data, device, u=u)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
        t += 0.01
