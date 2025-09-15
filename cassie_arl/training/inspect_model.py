import mujoco as mj
from pathlib import Path
import jax
from jax import numpy as jnp

from cassie_arl.config.cassie_consts import *


script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir / ".." / "models" / "scene.xml"

model = mj.MjModel.from_xml_path(CASSIE_SCENE_XML.as_posix())

# floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'floor')
# print(f"ID of 'floor' geom: {floor_id}")

# cassie_pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'cassie-pelvis')
# print(f"ID of 'cassie-pelvis' geom: {cassie_pelvis_id}")


# print(model.jnt_range[1:].T)

# print(model.actuator_ctrlrange.T)

for i, (low, high) in enumerate(model.actuator_ctrlrange):
    print(i, model.actuator(i).name, low, high)

# # Get Cassie's home pose
# home_pose = jnp.array(model.keyframe("home").qpos)
# home_motor_pose = home_pose[MOTOR_IDX]
# print("Cassie's home motor pose:")
# print(home_motor_pose)

# # print(model.jnt_range[1:].T)
# for i in range(model.nu):  # model.nu = number of actuators
#     name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
#     trnid = model.actuator_trnid[i]  # (joint_id, second_id)
#     print(i, name, trnid)
# print("\n")

# for i in range(model.njnt):
#     name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
#     print(i, name, model.jnt_type[i], model.jnt_range[i])


print(model.jnt_range[jnp.array(JntRangeIdx.MOTORS)])
print(JntRangeIdx.MOTORS)
