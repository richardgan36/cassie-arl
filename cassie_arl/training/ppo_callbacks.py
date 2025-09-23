"""Callback functions passed into PPO training loop."""
from absl import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco as mj
import jax
import numpy as np
import cv2
from brax.training import types

from cassie_arl.rl_env.math_utils import vec_xy_world_to_base
from cassie_arl.config.cassie_consts import *


logging.set_verbosity(logging.INFO)


class ProgressCallback:
    """Callable progress callback for Brax PPO training loop."""
    def __init__(
            self,
            training_params: dict,
            script_dir: Path, train_id: str,
            save_plot: bool = True
        ):
        self.training_params = training_params
        self.script_dir = script_dir
        self.train_id = train_id
        self.save_plot = save_plot
        self.x_data = []
        self.y_data = []
        self.y_dataerr = []
        self.times = [datetime.now()]

    def __call__(self, num_steps: int, metrics: types.Metrics | dict):
        self.times.append(datetime.now())
        self.x_data.append(num_steps)
        self.y_data.append(metrics["eval/episode_reward"])
        self.y_dataerr.append(metrics["eval/episode_reward_std"])

        print("")
        logging.info("--- Progress update ---")
        logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if len(self.times) == 2:
            time_to_jit = self.times[-1] - self.times[0]
            logging.info(f"Steps: {num_steps}, Reward: {self.y_data[-1]:.3f} ± {self.y_dataerr[-1]:.3f}")
            logging.info(f"Time to jit: {time_to_jit}")
        else:
            delta = self.times[-1] - self.times[-2]
            last_step = self.x_data[-2] if len(self.x_data) >= 2 else None
            logging.info(f"Steps: {num_steps}, Reward: {self.y_data[-1]:.3f} ± {self.y_dataerr[-1]:.3f}")
            logging.info(f"Time since last progress call (steps {last_step} -> {num_steps}): {delta}")
        print("-----------------")

        plt.clf()  # Clear the current figure
        plt.errorbar(self.x_data, self.y_data, yerr=self.y_dataerr, color="blue")
        plt.xlim([0, self.training_params["num_timesteps"] * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.y_data[-1]:.3f}")
        
        if not self.save_plot:
            plt.pause(0.005)  # Small pause to update the figure
            return

        timestamp_day = datetime.now().strftime("%Y-%m-%d")
        save_path = self.script_dir / "progress" / self.train_id / f"progress_{timestamp_day}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")




class VisualizePolicyCallback:
    """Callable visualization callback for Brax PPO training loop."""
    def __init__(self, env, jit_reset, jit_step, script_dir: Path, train_id: str):
        self.env = env
        self.jit_reset = jit_reset
        self.jit_step = jit_step
        self.script_dir = script_dir
        self.train_id = train_id

    def __call__(self, current_step: int, make_policy, params):
        try:
            print("")
            logging.info("--- Visualization update ---")
            logging.info(f"Generating rollout video at step {current_step}")
            inference_fn = make_policy(params, deterministic=True)
            rng = jax.random.PRNGKey(int(current_step+1) & 0xFFFFFFFF)
            state = self.jit_reset(rng)

            traj = []
            reward_info_list = []  # Store all reward components
            com_info_list = []  # Store COM, distance, contacts, reward
            lift_foot_info_list = []  # Store foot heights and reward
            cumulative_reward = 0.0
            
            for i in range(self.env._config.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = inference_fn(state.obs, act_rng)
                state = self.jit_step(state, ctrl)
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
                vec_to_support_world = self.env._vector_com_to_support(state.data)
                vec_to_support_base = vec_xy_world_to_base(vec_to_support_world, state.data.qpos[QPosIdx.BASE_QUAT])
                dist_to_support = np.linalg.norm(np.array(vec_to_support_world))
                left_contact = bool(self.env._is_in_contact_with_ground(state.data, self.env._left_foot_gid))
                right_contact = bool(self.env._is_in_contact_with_ground(state.data, self.env._right_foot_gid))
                com_info_list.append({
                    "dist": dist_to_support,
                    "com_outside_support":  dist_to_support > self.env._config.com_outside_support_threshold,
                    "left_contact": left_contact,
                    "right_contact": right_contact,
                    "vec_to_support": np.array(vec_to_support_base),
                })

                # ---- Foot lift info ----
                # Use body positions (xpos) for foot z coordinates;
                left_foot_z = float(np.array(state.data.xpos[self.env._left_foot_id, 2])) - FOOT_OFFSET
                right_foot_z = float(np.array(state.data.xpos[self.env._right_foot_id, 2])) - FOOT_OFFSET
                lift_foot_info_list.append({
                    "lift_foot_given": bool(state.info["lift_foot_given"]),
                    "left_foot_z": left_foot_z,
                    "right_foot_z": right_foot_z,
                    "left_tarsus_z": float(np.array(state.data.xpos[self.env._left_tarsus_id, 2])),
                    "right_tarsus_z": float(np.array(state.data.xpos[self.env._right_tarsus_id, 2])),
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
            frames = self.env.render(traj, camera="track", scene_option=scene_option, width=640*2, height=480)

            # Overlay COM sphere and text
            frames_overlay = []
            # Get reward scale config for showing raw and weighted values
            reward_scales = {k: float(v) for k, v in self.env._config.reward_config.scales.items()}
            
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
                    f"Left tarsus z: {lift_foot_info['left_tarsus_z']:.3f} m",
                    f"Right tarsus z: {lift_foot_info['right_tarsus_z']:.3f} m",
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
            fps = float(1.0 / getattr(self.env, "dt", 0.02))
            ani_save_dir = self.script_dir / "simulation" / f"{self.train_id}_3" / "test"
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
            logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info("----------------")

        except Exception as e:
            logging.exception(f"Failed to generate/save rollout video at step {current_step}: {e}")

