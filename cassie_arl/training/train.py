# Python built-in packages
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
from pathlib import Path

# Third-party packages
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from flax import struct
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from mujoco_playground._src.gait import draw_joystick_command


script_dir = Path(__file__).parent.resolve()

plt.ion()  # Interactive mode


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.clf()  # Clear the current figure
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.pause(0.005)  # Small pause to update the figure


env_name = 'BerkeleyHumanoidJoystickFlatTerrain'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

# # --- Shorten training for testing ---
# ppo_training_params["num_evals"] = 20
# ppo_training_params["episode_length"] = 2
# ppo_training_params["num_envs"] = 64
# ppo_training_params["num_minibatches"] = 4
ppo_training_params["num_timesteps"] = 500_000

print("[INFO] PPO training parameters:")
print(ppo_training_params)

save_ckpt_dir = script_dir / "checkpoints" / f"{env_name}_ppo"
print(f"[INFO] Checkpoints will be saved to {save_ckpt_dir}")

restore_ckpt_path = save_ckpt_dir / "000075694080"

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    save_checkpoint_path=str(save_ckpt_dir),
    restore_checkpoint_path=str(restore_ckpt_path)
)

# Start training
print("[INFO] Starting training")
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

env = registry.load(env_name)
eval_env = registry.load(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

x_vel = 1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
yaw_vel = 0.0  #@param {type: "number"}
command = jp.array([x_vel, y_vel, yaw_vel])

phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase = jp.array([0, jp.pi])

print("[INFO] Generating rollout")
for j in range(1):
    print(f"episode {j}")
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        if state.done:
            break
        state.info["command"] = command
        rollout.append(state)

        xyz = np.array(state.data.xpos[eval_env.mj_model.body("torso").id])
        xyz += np.array([0, 0.0, 0])
        x_axis = state.data.xmat[eval_env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=np.linalg.norm(state.info["command"]),
            )
        )


render_every = 1
fps = 1.0 / eval_env.dt / render_every
print(f"fps: {fps}")
traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]

print("[INFO] Rendering video at {} fps".format(fps))

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

frames = eval_env.render(
    traj,
    camera="track",
    scene_option=scene_option,
    width=640*2,
    height=480,
    modify_scene_fns=mod_fns,
)

# ---- Matplotlib animation setup ----
fig, ax = plt.subplots()
ax.axis('off')  # hide axes

# Display the first frame
im = ax.imshow(frames[0])

# Update function for animation
def update(frame):
    im.set_data(frame)
    return [im]

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=1000/fps,  # milliseconds per frame
    blit=True
)

# ---- Save as MP4 ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ani.save(f"{script_dir}/simulation-{timestamp}.mp4", writer='ffmpeg', fps=fps)
print("[INFO] Saved video to simulation.mp4")

