import mujoco
from mujoco import viewer
from pathlib import Path
import numpy as np
import time

# Path to your XML
scene_xml_path = Path(__file__).parents[1] / "models/scene.xml"

# Load model and create data object
model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
data = mujoco.MjData(model)

low, high = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]

with viewer.launch_passive(model, data) as v:
    simstart = time.time()
    while v.is_running():
        t = data.time
        # data.ctrl[0:5] = A * np.sin(2 * np.pi * 0.5 * t)
        # data.ctrl[5:10] = A * np.cos(2 * np.pi * 0.5 * t)

        data.ctrl[:] = np.random.uniform(low, high)

        mujoco.mj_step(model, data)

        # keep simulation synced to real time
        wall_time = time.time() - simstart
        sim_time = data.time
        if sim_time > wall_time:
            time.sleep(sim_time - wall_time)

        v.sync(state_only=True)
