import sys
import time
import mujoco
import mujoco.viewer

if len(sys.argv) < 2:
    raise SystemExit("Usage: mjpython view_model.py path/to/model.xml")

model_path = sys.argv[1]
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
