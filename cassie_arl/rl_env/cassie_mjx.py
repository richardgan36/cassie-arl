import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx


#########################################
### This file is a placeholder interface and is yet to be implemented
#########################################


class CassieSim:
    def __init__(self, model_file):
        self.model = mujoco.MjModel.from_xml_path(model_file)
        self.data = mjx.make_data(self.model)

        # Dimensions from model
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nu = self.model.nu
        self.nbody = self.model.nbody
        self.ngeom = self.model.ngeom

    def reset(self):
        self.data = mjx.make_data(self.model)
        return self.data

    def step(self, ctrl):
        # ctrl is a (nu,) array
        self.data = mjx.step(self.model, self.data, ctrl)
        return self.data

    def qpos(self):
        return self.data.qpos

    def qvel(self):
        return self.data.qvel

    def time(self):
        return self.data.time

    def set_state(self, qpos, qvel):
        self.data = self.data.replace(qpos=qpos, qvel=qvel)

    def apply_force(self, force, body_id):
        # MJX doesn't expose mj_applyFT directly yet; you’d model this as external forces.
        pass
