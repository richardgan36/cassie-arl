# Copyright 2025 Richard Gan
#
# This file is adapted from Mujoco Playground's domain randomization,
# and is therefore under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Cassie environment.

This module provides domain randomization capabilities for the following parameters:
friction, mass distribution, actuator properties, and initial poses.
"""

import jax
from mujoco import mjx


# Body IDs (must match MJCF model structure)
FLOOR_GEOM_ID = 0
PELVIS_BODY_ID = 1

# Number of actuated joints (motors) in Cassie
NUM_MOTORS = 10

# First actuated DoF index (after 6 DoF freejoint)
FIRST_MOTOR_DOF = 6


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Apply domain randomization to Cassie's physical parameters.
    
    This function randomizes various physical properties to improve sim-to-real
    transfer and policy robustness. Parameters randomized include:
    - Floor friction coefficient
    - Joint friction losses (damping)
    - Joint armature (motor inertia)
    - Body masses (all links + pelvis payload)
    - Initial joint positions (qpos0)
    
    The randomization is vectorized to support parallel environment execution
    in Brax.
    
    Args:
        model: The MJX model to randomize
        rng: JAX random key for sampling
        
    Returns:
        Tuple of (randomized_model, in_axes) where:
            - randomized_model: Model with randomized parameters
            - in_axes: Tree structure indicating which axes to vectorize over
    """
    
    @jax.vmap
    def rand_dynamics(rng):
        """Vectorized randomization of dynamics parameters.
        
        Each parallel environment gets different randomized parameters sampled
        from uniform distributions.
        """
        
        # Floor friction: U(0.6, 1.2)
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
            jax.random.uniform(key, minval=0.6, maxval=1.2)
        )
        
        # Joint friction losses: * U(0.8, 1.2) [DISABLED]
        # Only randomize actuated joints (skip freejoint DoFs).
        # rng, key = jax.random.split(rng)
        # motor_dofs = slice(FIRST_MOTOR_DOF, FIRST_MOTOR_DOF + NUM_MOTORS)
        # frictionloss = model.dof_frictionloss[motor_dofs] * jax.random.uniform(
        #     key, shape=(NUM_MOTORS,), minval=0.8, maxval=1.2
        # )
        # dof_frictionloss = model.dof_frictionloss.at[motor_dofs].set(frictionloss)
        dof_frictionloss = model.dof_frictionloss  # No randomization
        
        # Joint armature: * U(1.0, 1.1) [DISABLED]
        # rng, key = jax.random.split(rng)
        # motor_dofs = slice(FIRST_MOTOR_DOF, FIRST_MOTOR_DOF + NUM_MOTORS)
        # armature = model.dof_armature[motor_dofs] * jax.random.uniform(
        #     key, shape=(NUM_MOTORS,), minval=1.0, maxval=1.1
        # )
        # dof_armature = model.dof_armature.at[motor_dofs].set(armature)
        dof_armature = model.dof_armature  # No randomization
        
        # Pelvis mass: * U(0.8, 1.2)
        # ±20% variation of pelvis mass to simulate mass uncertainty and payloads.
        rng, key = jax.random.split(rng)
        pelvis_mass_multiplier = jax.random.uniform(key, minval=0.8, maxval=1.2)
        body_mass = model.body_mass.at[PELVIS_BODY_ID].set(
            model.body_mass[PELVIS_BODY_ID] * pelvis_mass_multiplier
        )
        
        # Initial joint positions: + U(-0.05, 0.05) rad [DISABLED]
        # rng, key = jax.random.split(rng)
        # qpos0 = model.qpos0
        # # Cassie has 7 qpos for freejoint (pos + quat) + actuated joints
        # # Perturb all joints after the freejoint
        # num_jnt_qpos = len(qpos0) - 7  # All joints after freejoint
        # qpos0 = qpos0.at[7:].set(
        #     qpos0[7:] + jax.random.uniform(
        #         key, shape=(num_jnt_qpos,), minval=-0.05, maxval=0.05
        #     )
        # )
        qpos0 = model.qpos0  # No randomization
        
        return (
            geom_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
        )
    
    # Apply vectorized randomization
    (
        friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = rand_dynamics(rng)
    
    # Create in_axes specification for vmap
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "body_mass": 0,
        "qpos0": 0,
    })
    
    # Create randomized model with new parameters
    model = model.tree_replace({
        "geom_friction": friction,
        "dof_frictionloss": frictionloss,
        "dof_armature": armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
    })
    
    return model, in_axes
