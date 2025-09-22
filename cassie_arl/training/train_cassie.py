# Python built-in packages
from datetime import datetime
import functools
from pathlib import Path
import os

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
from cassie_arl.rl_env.math_utils import vec_xy_world_to_base
from cassie_arl.config.cassie_consts import *


os.environ["ABSL_LOG_PREFIX"] = "1"
script_dir = Path(__file__).parent.resolve()

logging.set_verbosity(logging.INFO)

train_id = "cassie_obs38"  # Observation space is 38D

network_factory_params = {
    "policy_hidden_layer_sizes": (512, 256, 128),
    "policy_obs_key": "state",
    "value_hidden_layer_sizes": (512, 256, 128),
    "value_obs_key": "privileged_state",
}

ppo_training_params = {
    'action_repeat': 1,
    'batch_size': 128,
    'clipping_epsilon': 0.2,
    'discounting': 0.98,  # TODO: Used to be 0.97. Change back?
    'entropy_cost': 0.005,
    'episode_length': 512,
    'learning_rate': 0.0003,
    'max_grad_norm': 1.0,
    'normalize_observations': True,
    'num_envs': 2048,
    'num_evals': 20,
    'num_minibatches': 320,
    'num_resets_per_eval': 1,
    'num_timesteps': 200_000_000,
    'num_updates_per_batch': 4,
    'reward_scaling': 2.0,
    'unroll_length': 20,
    'restore_value_fn': True,
}


def progress(num_steps: int, metrics: types.Metrics | dict):
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

    # logging.info(f"steps: {num_steps}, reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
    # logging.info(f"time since last progress call: {times[-1] - times[-2]}")

    # Save the figure with date (day only, no time)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    save_path = script_dir / "progress" / train_id / f"progress_{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")

    if len(times) == 2:
        time_to_jit = times[-1] - times[0]
        logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
        logging.info(f"Time to jit: {time_to_jit}")
    else:
        delta = times[-1] - times[-2]
        last_step = x_data[-2] if len(x_data) >= 2 else None
        logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
        logging.info(f"Time since last progress call (steps {last_step} -> {num_steps}): {delta}")


# def progress(num_steps: int, metrics: types.Metrics | dict):
#     # Lazily initialise an interactive figure and axes on first call.  Creating
#     # the figure once and re-using it avoids backend race conditions that can
#     # produce an empty/blank window on the first few updates.
#     if not hasattr(progress, "_inited"):
#         plt.ion()
#         progress._fig, progress._ax = plt.subplots()
#         progress._line, = progress._ax.plot([], [], color="blue")
#         # store the errorbar artists so we can remove them before drawing new ones
#         progress._eb = None
#         progress._ax.set_xlabel("# environment steps")
#         progress._ax.set_ylabel("reward per episode")
#         progress._inited = True

#     times.append(datetime.now())
#     x_data.append(num_steps)
#     y = float(metrics.get("eval/episode_reward", np.nan))
#     yerr = float(metrics.get("eval/episode_reward_std", 0.0))
#     y_data.append(y)
#     y_dataerr.append(yerr)

#     # Update line data and errorbars without recreating the figure.
#     progress._line.set_data(x_data, y_data)
#     # remove previous errorbar artists
#     if progress._eb is not None:
#         try:
#             for coll in progress._eb[2] if len(progress._eb) > 2 else []:
#                 coll.remove()
#         except Exception:
#             pass
#     progress._eb = progress._ax.errorbar(x_data, y_data, yerr=y_dataerr, color="blue", fmt="none")

#     # Keep x axis fixed to full training horizon for consistency.
#     progress._ax.set_xlim([0, ppo_training_params["num_timesteps"] * 1.25])
#     # Autoscale y for visibility.
#     progress._ax.relim()
#     progress._ax.autoscale_view(scalex=False, scaley=True)

#     # Force a draw and flush GUI events so the window updates immediately.
#     progress._fig.canvas.draw()
#     progress._fig.canvas.flush_events()
#     # plt.pause(0.001)

#     # Save the figure with date (day only, no time)
#     timestamp = datetime.now().strftime("%Y-%m-%d")
#     save_path = script_dir / "progress" / train_id / f"progress_{timestamp}.png"
#     save_path.parent.mkdir(parents=True, exist_ok=True)
#     progress._fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

#     # Log timing information:
#     # - On the first progress call, report time to jit (time since script start).
#     # - On subsequent calls, report time since last progress call and include
#     #   the previous and current step numbers.
#     if len(times) == 2:
#         time_to_jit = times[-1] - times[0]
#         logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
#         logging.info(f"Time to jit: {time_to_jit}")
#     else:
#         delta = times[-1] - times[-2]
#         last_step = x_data[-2] if len(x_data) >= 2 else None
#         logging.info(f"Steps: {num_steps}, Reward: {y_data[-1]:.3f} ± {y_dataerr[-1]:.3f}")
#         logging.info(f"Time since last progress call (steps {last_step} -> {num_steps}): {delta}")


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
        reward_info_list = []  # Store all reward components
        com_info_list = []  # store COM, distance, contacts, reward
        lift_foot_info_list = []  # store foot heights and reward
        cumulative_reward = 0.0
        
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            if bool(state.done):
                break
            
            # Track rewards
            step_reward = float(state.reward)
            cumulative_reward += step_reward
            
            # Store reward components for visualization
            reward_info_list.append({
                "components": {k: float(v) for k, v in state.info["reward_components"].items()},
                "step_reward": step_reward,
                "cumulative_reward": cumulative_reward
            })
            
            traj.append(state)

            # ---- COM info ----
            vec_to_support_world = env._vector_com_to_support(state.data)
            vec_to_support_base = vec_xy_world_to_base(vec_to_support_world, state.data.qpos[QPosIdx.BASE_QUAT])
            dist_to_support = np.linalg.norm(np.array(vec_to_support_world))
            left_contact = bool(env._is_in_contact_with_ground(state.data, env._left_foot_gid))
            right_contact = bool(env._is_in_contact_with_ground(state.data, env._right_foot_gid))
            com_info_list.append({
                "dist": dist_to_support,
                "com_outside_support":  dist_to_support > env._config.com_outside_support_threshold,
                "left_contact": left_contact,
                "right_contact": right_contact,
                "vec_to_support": np.array(vec_to_support_base),
            })

            # ---- Foot lift info ----
            # Use body positions (xpos) for foot z coordinates;
            left_foot_z = float(np.array(state.data.xpos[env._left_foot_id, 2])) - FOOT_OFFSET
            right_foot_z = float(np.array(state.data.xpos[env._right_foot_id, 2])) - FOOT_OFFSET
            lift_foot_info_list.append({
                "lift_foot_given": bool(state.info["lift_foot_given"]),
                "left_foot_z": left_foot_z,
                "right_foot_z": right_foot_z,
            })

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
        # Get reward scale config for showing raw and weighted values
        reward_scales = {k: float(v) for k, v in env._config.reward_config.scales.items()}
        
        for f_idx, frame in enumerate(frames):
            frame_rgb = np.array(frame).copy()
            
            # COM info
            com_info = com_info_list[f_idx]
            com_outside_support = com_info['com_outside_support']
            vec_to_support = com_info['vec_to_support']
            com_vec_x = vec_to_support[0]
            com_vec_y = vec_to_support[1]

            # Foot info
            lift_foot_info = lift_foot_info_list[f_idx]
            left_foot_z = lift_foot_info['left_foot_z']
            right_foot_z = lift_foot_info['right_foot_z']
            lift_foot_given = lift_foot_info['lift_foot_given']
            
            # Reward info
            reward_info = reward_info_list[f_idx]
            components = reward_info["components"]
            step_reward = reward_info["step_reward"]
            cumulative_reward = reward_info["cumulative_reward"]

            # Left side text - standard info
            left_text_lines = [
                f"COM outside support: {com_outside_support}",
                f"COM->Support dist: {com_info['dist']:.3f} m",
                f"Vec to support (base): ({com_vec_x:.3f}, {com_vec_y:.3f}) m",
                f"L contact: {com_info['left_contact']}, R contact: {com_info['right_contact']}",
                f"Left foot z: {left_foot_z:.3f} m",
                f"Right foot z: {right_foot_z:.3f} m",
                f"Lift foot reward given: {lift_foot_given}",
            ]
            
            # Right side text - reward components
            right_text_lines = [
                f"Step Reward: {step_reward:.4f}",
                f"Cumulative Reward: {cumulative_reward:.4f}",
                "-----------------------------",
                "Component           Raw Value | Weighted",
            ]
            
            # Add each reward component with its raw and weighted values
            for component_name, weighted_value in components.items():
                # Get the raw value by dividing by the scale factor
                scale = reward_scales.get(component_name, 1.0)
                raw_value = weighted_value / scale if scale != 0 else 0
                
                # Format with alignment for better readability
                # Right-align numbers for easier comparison
                component_name_padded = f"{component_name}:".ljust(18)
                component_line = f"{component_name_padded} {raw_value:>8.4f} | {weighted_value:>8.4f}"
                right_text_lines.append(component_line)
                
            # Add left side text
            for idx, line in enumerate(left_text_lines):
                cv2.putText(frame_rgb, line, (10, 30 + idx*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
            # Add right side text - position on right side of frame
            frame_width = frame_rgb.shape[1]
            right_text_x = frame_width - 500  # Position from right side
            
            for idx, line in enumerate(right_text_lines):
                # Use different colors for positive/negative rewards
                # Note: OpenCV uses BGR format (not RGB)
                color = (255, 255, 255)  # Default white
                if idx >= 4:  # For reward component lines
                    component_name = line.split(':')[0]
                    if component_name in components:
                        if components[component_name] > 0:
                            color = (0, 255, 0)  # Green for positive rewards
                        elif components[component_name] < 0:
                            color = (255, 100, 100)  # Red (pinkish) for negative rewards
                            
                cv2.putText(frame_rgb, line, (right_text_x, 30 + idx*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
ppo_training_params["num_evals"] = 2
ppo_training_params["episode_length"] = 5
ppo_training_params["num_envs"] = 1
ppo_training_params["num_minibatches"] = 4
ppo_training_params["batch_size"] = 2
ppo_training_params["unroll_length"] = 8
ppo_training_params["num_timesteps"] = 1000

logging.info("PPO training parameters:")
logging.info(ppo_training_params)

save_ckpt_dir = script_dir / "checkpoints" / f"{train_id}_2"
restore_ckpt_path = script_dir / "checkpoints" / f"{train_id}_2" / "000021299200"

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

