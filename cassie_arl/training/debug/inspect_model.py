import mujoco as mj
from mujoco import mjx
from pathlib import Path
import jax
from jax import numpy as jnp

from cassie_arl.cassie_env.cassie_consts import *


script_dir = Path(__file__).parent
CASSIE_SCENE_XML = script_dir / ".." / "models" / "scene.xml"

model = mj.MjModel.from_xml_path(CASSIE_SCENE_XML.as_posix())
data = mj.MjData(model)


mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'floor')
# print(model.geom('floor').id)
# print(f"ID of 'floor' geom: {floor_id}")

# cassie_pelvis_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'cassie-pelvis')
# print(f"ID of 'cassie-pelvis' geom: {cassie_pelvis_id}")


# print(model.jnt_range[1:].T)

# print(model.actuator_ctrlrange.T)

# for i, (low, high) in enumerate(model.actuator_ctrlrange):
#     print(i, model.actuator(i).name, low, high)

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


# print(model.jnt_range[jnp.array(JntRangeIdx.MOTORS)])
# print(JntRangeIdx.MOTORS)


# Get geom IDs for left and right foot collision geoms
# left_ids = [i for i, name in enumerate(model.geom_names) if "collision-left" in model.geom_classname[i]]
# right_ids = [i for i, name in enumerate(model.geom_names) if "collision-right" in model.geom_classname[i]]

# for i, name in enumerate(model.geom_names):
#     if "collision-left" in model.geom_classname[i] or "collision-right" in model.geom_classname[i]:
#         print(f"ID: {i}, Geom name: {name}")




# try:
#     for i in range(1000):
#         print(i, model.body(i).name)
# except:
#     exit(0)

# print(type(mjx_data._impl.contact.pos))

def geoms_of_body(model, body_id):
    start = model.body_geomadr[body_id]
    count = model.body_geomnum[body_id]
    geom_ids = jnp.arange(start, start + count)
    return geom_ids

# print(geoms_of_body(model, 25))

# for i in range(model.ngeom):
#     name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
#     print(f"geom id={i}, name={name}")


# print(f"geom[0]: {mjx_data._impl.contact.geom[:, 0]}")
# print(f"geom1: {mjx_data._impl.contact.geom1}")

# mj.mj_step(model, data)
# print(data.contact)

# print(mjx_data._impl.contact.frame)

print(model.body("com_marker").id)