# Python built-in packages
from datetime import datetime
import functools
from pathlib import Path

# Third-party packages
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training import types
import jax
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import mujoco as mj
from mujoco import mjx
from absl import logging
from mujoco_playground import wrapper
import cv2

from cassie_arl.rl_env.cassie_env import CassieEnv, default_config
from cassie_arl.rl_env.cassie_domain_randomizer import domain_randomize


script_dir = Path(__file__).parent.resolve()

logging.set_verbosity(logging.INFO)

train_id = "cassie_ppo_obs34"  # Observation space is 34D

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
    'discounting': 0.97,  # TODO: Consider increasing
    'entropy_cost': 0.005,
    'episode_length': 512,
    'learning_rate': 0.0003,
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 8192,
    'num_evals': 20,
    'num_minibatches': 640,
    'num_resets_per_eval': 1,
    'num_timesteps': 300_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 1.0,
    'unroll_length': 20,
    'restore_value_fn': True
}


# def progress(num_steps: int, metrics: types.Metrics | dict):
#     times.append(datetime.now())
#     x_data.append(num_steps)
#     y_data.append(metrics["eval/episode_reward"])
#     y_dataerr.append(metrics["eval/episode_reward_std"])

#     plt.clf()  # Clear the current figure
#     plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
#     plt.xlim([0, ppo_training_params["num_timesteps"] * 1.25])
#     plt.xlabel("# environment steps")
#     plt.ylabel("reward per episode")
#     plt.title(f"y={y_data[-1]:.3f}")
#     plt.pause(0.005)  # Small pause to update the figure

#     logging.info(f"steps: {num_steps}, reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
#     logging.info(f"time since last progress call: {times[-1] - times[-2]}")

#     # save_path = script_dir / "progress" / "cassie_ppo" / "progress-9-13.png"
#     # save_path.parent.mkdir(parents=True, exist_ok=True)
#     # plt.savefig(str(save_path), dpi=150, bbox_inches="tight")


def progress(num_steps: int, metrics: types.Metrics | dict):
    # Lazily initialise an interactive figure and axes on first call.  Creating
    # the figure once and re-using it avoids backend race conditions that can
    # produce an empty/blank window on the first few updates.
    if not hasattr(progress, "_inited"):
        plt.ion()
        progress._fig, progress._ax = plt.subplots()
        progress._line, = progress._ax.plot([], [], color="blue")
        # store the errorbar artists so we can remove them before drawing new ones
        progress._eb = None
        progress._ax.set_xlabel("# environment steps")
        progress._ax.set_ylabel("reward per episode")
        progress._inited = True

    times.append(datetime.now())
    x_data.append(num_steps)
    y = float(metrics.get("eval/episode_reward", np.nan))
    yerr = float(metrics.get("eval/episode_reward_std", 0.0))
    y_data.append(y)
    y_dataerr.append(yerr)

    # Update line data and errorbars without recreating the figure.
    progress._line.set_data(x_data, y_data)
    # remove previous errorbar artists
    if progress._eb is not None:
        try:
            for coll in progress._eb[2] if len(progress._eb) > 2 else []:
                coll.remove()
        except Exception:
            pass
    progress._eb = progress._ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue", fmt="none")

    # Keep x axis fixed to full training horizon for consistency.
    progress._ax.set_xlim([0, ppo_training_params["num_timesteps"] * 1.25])
    # Autoscale y for visibility.
    progress._ax.relim()
    progress._ax.autoscale_view(scalex=False, scaley=True)

    # Force a draw and flush GUI events so the window updates immediately.
    progress._fig.canvas.draw()
    progress._fig.canvas.flush_events()
    plt.pause(0.001)

    # Log timing information:
    # - On the first progress call, report time to jit (time since script start).
    # - On subsequent calls, report time since last progress call and include
    #   the previous and current step numbers.
    if len(times) == 2:
        time_to_jit = times[-1] - times[0]
        logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
        logging.info(f"Time to jit: {time_to_jit}")
    else:
        delta = times[-1] - times[-2]
        last_step = x_data[-2] if len(x_data) >= 2 else None
        logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
        logging.info(f"Time since last progress call (steps {last_step} -> {num_steps}): {delta}")


# def visualize_policy(
#         current_step: int,
#         make_policy: types.Policy,
#         params: types.Params
#     ):
#     try:
#         logging.info(f"Generating rollout video at step {current_step}")

#         # Build deterministic inference function from the provided factory.
#         inference_fn = make_policy(params, deterministic=True)

#         # Seed with current_step so videos vary across checkpoints.
#         rng = jax.random.PRNGKey(int(current_step) & 0xFFFFFFFF)
#         state = jit_reset(rng)

#         # Rollout using current policy
#         traj = []
#         logging.info("Starting rollout")
#         for i in range(env_cfg.episode_length):
#             act_rng, rng = jax.random.split(rng)
#             ctrl, _ = inference_fn(state.obs, act_rng)
#             state = jit_step(state, ctrl)
#             if bool(state.done):
#                 break
#             traj.append(state)

#         logging.info(f"Rollout finished after {len(traj)} steps")

#         if len(traj) == 0:
#             logging.warning("No frames collected for rollout; skipping video save.")
#             return

#         # Rendering options
#         scene_option = mj.MjvOption()
#         scene_option.geomgroup[2] = True
#         scene_option.geomgroup[3] = False
#         scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
#         scene_option.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
#         scene_option.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = False

#         logging.info("Rendering rollout")
#         frames = env.render(
#             traj,
#             camera="track",
#             scene_option=scene_option,
#             width=640 * 2,
#             height=480,
#         )

#         # Save animation
#         ani_save_dir = script_dir / "simulation" / train_id
#         ani_save_dir.mkdir(parents=True, exist_ok=True)
#         fps = float(1.0 / getattr(env, "dt", 0.02))

#         fig, ax = plt.subplots()
#         ax.axis("off")
#         im = ax.imshow(np.asarray(frames[0]))

#         def update(frame):
#             im.set_data(np.asarray(frame))
#             return [im]

#         logging.info("Saving rollout video")

#         ani = animation.FuncAnimation(
#             fig,
#             update,
#             frames=frames,
#             interval=1000 / fps,
#             blit=True,
#         )

#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         ani_save_path = ani_save_dir / f"rollout_step{current_step}-{timestamp}.mp4"
#         ani.save(ani_save_path, writer="ffmpeg", fps=fps)
#         plt.close(fig)
#         logging.info(f"Saved rollout video to {ani_save_path}")
#     except Exception as e:
#         logging.exception(f"Failed to generate/save rollout video at step {current_step}: {e}")


def visualize_policy(current_step: int, make_policy, params):
    try:
        logging.info(f"Generating rollout video at step {current_step}")
        inference_fn = make_policy(params, deterministic=True)
        rng = jax.random.PRNGKey(int(current_step) & 0xFFFFFFFF)
        state = jit_reset(rng)

        traj = []
        com_info_list = []  # store COM, distance, contacts, reward
        logging.info("Starting rollout")
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            if bool(state.done):
                break
            traj.append(state)

            # ---- COM info ----
            com = np.array(state.data.subtree_com[0])  # full 3D
            vec_to_support = env._vector_com_to_support(state.data)
            dist_to_support = np.linalg.norm(np.array(vec_to_support))
            left_contact = bool(env._is_in_contact_with_ground(state.data, env._left_foot_gid))
            right_contact = bool(env._is_in_contact_with_ground(state.data, env._right_foot_gid))
            com_cost = float(state.info["reward_components"]["com_outside_support"])
            com_info_list.append({
                "com": com,
                "dist": dist_to_support,
                "left_contact": left_contact,
                "right_contact": right_contact,
                "cost": com_cost,
                # "geom1": np.array(state.data._impl.contact.geom[:, 0]),
                # "geom2": np.array(state.data._impl.contact.geom[:, 1]),
            })

            # # Filter the contacts by distance within a tolerance
            # tol = 0.001  # 1 mm tolerance
            # dists = state.data._impl.contact.dist
            # valid_contact_indices = np.where(dists <= tol)[0]
            # valid_dists = dists[valid_contact_indices]
            # valid_geom1 = state.data._impl.contact.geom[valid_contact_indices, 0]
            # valid_geom2 = state.data._impl.contact.geom[valid_contact_indices, 1]

            # logging.info(f"Geom1: {valid_geom1}\n Geom2: {valid_geom2}\n Distances: {valid_dists}\n\n")

        if len(traj) == 0:
            logging.warning("No frames collected; skipping.")
            return

        # Render frames
        scene_option = mj.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = False
        frames = env.render(traj, camera="track", scene_option=scene_option, width=640*2, height=480)

        # Overlay COM sphere and text
        frames_overlay = []
        for f_idx, frame in enumerate(frames):
            frame_rgb = np.array(frame).copy()
            info = com_info_list[f_idx]

            com = info['com']

            # Add text
            text_lines = [
                f"COM: ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})",
                f"COM->Support dist: {info['dist']:.3f} m",
                f"L_contact: {info['left_contact']}, R_contact: {info['right_contact']}",
                f"COM cost: {info['cost']:.3f}",
            ]
            for idx, line in enumerate(text_lines):
                cv2.putText(frame_rgb, line, (10, 30 + idx*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            frames_overlay.append(frame_rgb)

        # Save video
        fps = float(1.0 / getattr(env, "dt", 0.02))
        ani_save_dir = script_dir / "simulation" / train_id / "test"
        ani_save_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()
        ax.axis("off")
        im = ax.imshow(frames_overlay[0])

        def update(frame):
            im.set_data(frame)
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=frames_overlay, interval=1000/fps, blit=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ani_save_path = ani_save_dir / f"rollout_step{current_step}-{timestamp}.mp4"
        ani.save(ani_save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        logging.info(f"Saved rollout video to {ani_save_path}")

    except Exception as e:
        logging.exception(f"Failed to generate/save rollout video at step {current_step}: {e}")

env = CassieEnv()
env_cfg = default_config()
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

jit_step = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

# randomizer = domain_randomize
randomizer = None

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **dict(network_factory_params)
)

# # --- Shorten training for testing ---
ppo_training_params["num_evals"] = 5
ppo_training_params["episode_length"] = 5
ppo_training_params["num_envs"] = 1
ppo_training_params["num_minibatches"] = 4
ppo_training_params["batch_size"] = 2
ppo_training_params["num_timesteps"] = 99

logging.info("PPO training parameters:")
logging.info(ppo_training_params)

save_ckpt_dir = script_dir / "checkpoints" / train_id 
restore_ckpt_path = script_dir / "checkpoints" / train_id / "000143032320"

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
    policy_params_fn=visualize_policy,
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

logging.info(f"time to train: {times[-1] - times[1]}")

