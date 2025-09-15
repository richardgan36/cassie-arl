# Python built-in packages
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
from pathlib import Path

# Third-party packages
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import mujoco
from mujoco import mjx
import numpy as np
from absl import logging

from mujoco_playground import wrapper

from cassie_arl.rl_env.my_cassie_env import CassieEnv, default_config
from cassie_arl.rl_env.cassie_domain_randomizer import domain_randomize


script_dir = Path(__file__).parent.resolve()

# plt.ion()  # Interactive mode
logging.set_verbosity(logging.INFO)

network_factory_params = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    "policy_obs_key": "state",
    "value_hidden_layer_sizes": (512, 256, 128),
    "value_obs_key": "privileged_state",
}

ppo_training_params = {
    'action_repeat': 1,
    'batch_size': 256,
    'clipping_epsilon': 0.2,
    'discounting': 0.97,
    'entropy_cost': 0.005,
    'episode_length': 512,
    'learning_rate': 0.0003,
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 8192,
    'num_evals': 20,
    'num_minibatches': 32,
    'num_resets_per_eval': 1,
    'num_timesteps': 300_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 1.0,
    'unroll_length': 20
}


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.clf()  # Clear the current figure
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.xlim([0, ppo_training_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    # plt.pause(0.005)  # Small pause to update the figure

    logging.info(f"steps: {num_steps}, reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
    logging.info(f"time since last progress call: {times[-1] - times[-2]}")

    # save_path = script_dir / "progress" / "cassie_ppo" / "progress-9-13.png"
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # plt.savefig(str(save_path), dpi=150, bbox_inches="tight")


env = CassieEnv()
env_cfg = default_config()
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

# randomizer = domain_randomize
randomizer = None

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **dict(network_factory_params)
)

# # --- Shorten training for testing ---
ppo_training_params["num_evals"] = 1
ppo_training_params["episode_length"] = 5
ppo_training_params["num_envs"] = 1
ppo_training_params["num_minibatches"] = 4
ppo_training_params["batch_size"] = 2
ppo_training_params["num_timesteps"] = 100

logging.info("PPO training parameters:")
logging.info(ppo_training_params)

train_id = "cassie_ppo_obs34"  # Observation space is 34D

save_ckpt_dir = script_dir / "checkpoints" / train_id 
restore_ckpt_path = script_dir / "checkpoints" / train_id / "000143032320"

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    # save_checkpoint_path=str(save_ckpt_dir),
    restore_checkpoint_path=str(restore_ckpt_path)
)

if "save_checkpoint_path" in train_fn.keywords:
    logging.info(f"Checkpoints will be saved to {train_fn.keywords['save_checkpoint_path']}")

# Start training
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=CassieEnv(),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
logging.info(f"time to jit: {times[1] - times[0]}")
logging.info(f"time to train: {times[-1] - times[1]}")

eval_env = CassieEnv()
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# ---- Generate rollout and render to video ----
render_every = 1
rollout = []

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

# TODO: instead of just generating an animation at the very end of training,
#       do it throughout to get better progress info
logging.info("Generating rollout")
n_rollouts = 1   # number of random conditions to visualize
all_frames = []

for j in range(n_rollouts):
    logging.info(f"Rollout {j}")
    rng = jax.random.PRNGKey(j)
    state = jit_reset(rng)

    rollout = []
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)

        # --- Debugging: print action and reward components ---
        reward_components = state.info.get("reward_components", None)
        action = state.info.get("action", None)
        if reward_components is not None and action is not None:
            logging.info(f"Step {i}: action={np.asarray(action)}, rewards={ {k: float(np.asarray(v)) for k,v in reward_components.items()} }")
        # -----------------------------------------------------

        if bool(state.done):
            break
        rollout.append(state)

    traj = rollout[::render_every]

    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640*2,
        height=480,
    )
    all_frames.append(frames)

# ---- Save each rollout separately ----
ani_save_dir = script_dir / "simulation" / train_id
ani_save_dir.mkdir(parents=True, exist_ok=True)
fps = 1.0 / eval_env.dt / render_every
for j, frames in enumerate(all_frames):
    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000/fps,
        blit=True
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ani_save_path = ani_save_dir / f"rollout{j}-{timestamp}.mp4"
    ani.save(ani_save_path, writer="ffmpeg", fps=fps)
    logging.info(f"Saved rollout {j} to {ani_save_path}")
    plt.close(fig)
