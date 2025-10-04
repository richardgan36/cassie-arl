"""Callback functions passed into PPO training loop."""
import dataclasses
from absl import logging
from datetime import datetime, timedelta
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco as mj
import jax
import numpy as np
import cv2
from brax.training import types

from cassie_arl.config.cassie_consts import *


logging.set_verbosity(logging.INFO)


class ProgressCallback:
    """Callable progress callback for Brax PPO training loop."""
    def __init__(
            self,
            training_params: dict,
            script_dir: Path,
            train_id: str,
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
        save_path = self.script_dir / "progress" / self.train_id / f"progress.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")


class VisualizePolicyCallback:
    """Callable visualization callback for Brax PPO training loop."""
    def __init__(self, env, jit_reset, jit_step, script_dir: Path, train_id: str, run_every_n_calls: int = 1):
        self.env = env
        self.jit_reset = jit_reset
        self.jit_step = jit_step
        self.script_dir = script_dir
        self.train_id = train_id
        # How often to run visualization (in calls). 1 = every call, 2 = every 2nd call, etc.
        self.run_every_n_calls = run_every_n_calls
        self._call_count = 0

    def _reward_components_to_dict(self, rc) -> dict[str, float]:
        """Converts RewardComponents (flax.struct dataclass) or dict to a plain dict of floats.

        Safe to call regardless of whether rc is a dataclass or a mapping; falls back to best-effort.
        """
        try:
            # Handle flax.struct.dataclass (built on Python dataclasses)
            if dataclasses.is_dataclass(rc):
                out = {}
                for f in dataclasses.fields(rc):
                    val = getattr(rc, f.name)
                    # Convert JAX array to Python float for overlay
                    out[f.name] = float(np.array(val))
                return out
            # If it's already a mapping, convert values to float
            if isinstance(rc, dict):
                return {k: float(np.array(v)) for k, v in rc.items()}
        except Exception:
            pass
        # Last resort: try tree flatten with names unknown; return empty to avoid crashing overlays
        return {}

    def __call__(self, current_step: int, make_policy, params):
        try:
            # Increment call counter and optionally skip this visualization
            self._call_count += 1
            if (self._call_count % self.run_every_n_calls) != 0:
                logging.info(f"Skipping visualization call {self._call_count} (every {self.run_every_n_calls})")
                return
            start_time = time.time()
            print("")
            logging.info("--- Visualization update ---")
            logging.info(f"Generating rollout video at step {current_step}")
            inference_fn = make_policy(params, deterministic=True)
            rng = jax.random.PRNGKey(int(current_step+1) & 0xFFFFFFFF)
            state = self.jit_reset(rng)

            traj = []
            reward_info_list = []  # Store all reward components
            lift_foot_info_list = []  # Store foot heights and reward
            torque_history = []  # list of shape (10,) arrays
            angle_delta_history = []  # list of shape (10,) arrays from observations
            cumulative_reward = 0.0
            
            for i in range(self.env._config.episode_length):
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = inference_fn(state.obs, act_rng)

                torque = self.env._pd_control(
                    state.data,
                    ctrl,
                    self.env._p_gain,
                    self.env._d_gain,
                )

                state = self.jit_step(state, torque)
                if bool(state.done):
                    break

                # Store torque history for plotting later
                torque_history.append(np.array(torque))

                # Store joint angle deltas from observation for plotting later
                # Observation layout (see CassieEnv._get_obs):
                # [pelvis_height(1), pelvis_quat(4), motor_qpos_delta(10), ...]
                try:
                    motor_delta = np.array(state.obs[5:15])  # shape (10,)
                    if motor_delta.shape[0] == 10:
                        angle_delta_history.append(motor_delta)
                except Exception:
                    # If for any reason obs shape isn't as expected, skip recording
                    pass

                # Track rewards
                step_reward = float(state.reward)
                cumulative_reward += step_reward
                
                # Store reward components for visualization
                components_dict = self._reward_components_to_dict(state.info["reward_components"]) 
                reward_info_list.append({
                    "components": components_dict,
                    "step_reward": step_reward,
                    "cumulative_reward": cumulative_reward
                })
                
                traj.append(state)

                # ---- Foot lift info ----
                # Use body positions (xpos) for foot z coordinates;
                left_foot_z = float(np.array(state.data.xpos[self.env._left_foot_id, 2])) - FOOT_OFFSET
                right_foot_z = float(np.array(state.data.xpos[self.env._right_foot_id, 2])) - FOOT_OFFSET
                lift_foot_info_list.append({
                    # "lift_foot_given": bool(state.info["lift_foot_given"]),
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

            # Overlay diagnostic text
            frames_overlay = []
            # Get reward scale config for showing raw and weighted values
            reward_scales = {k: float(v) for k, v in self.env._config.reward_config.scales.items()}
            
            for f_idx, frame in enumerate(frames):
                frame_rgb = np.array(frame).copy()
                
                # Foot info
                lift_foot_info = lift_foot_info_list[f_idx]
                left_foot_z = lift_foot_info['left_foot_z']
                right_foot_z = lift_foot_info['right_foot_z']
                # lift_foot_given = lift_foot_info['lift_foot_given']
                
                # Reward info
                reward_info = reward_info_list[f_idx]
                components = reward_info["components"]
                step_reward = reward_info["step_reward"]
                cumulative_reward = reward_info["cumulative_reward"]

                # Left side text - standard info
                left_text_lines = [
                    f"Left foot z: {left_foot_z:.3f} m",
                    f"Right foot z: {right_foot_z:.3f} m",
                    # f"Lift foot reward given: {lift_foot_given}",
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
            # Pace playback by control step, not physics substep: real-time independent of sim_dt
            # The `env.dt` from the wrapper can be misleading. We use the unwrapped env's
            # `ctrl_dt` to get the true time per frame.
            dt_frame = float(self.env._config.ctrl_dt)
            fps = float(1.0 / dt_frame)
            ani_save_dir = self.script_dir / "simulation" / f"{self.train_id}" / "test"
            ani_save_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots()
            ax.axis("off")
            im = ax.imshow(frames_overlay[0])

            def update(frame):
                im.set_data(frame)
                return [im]

            ani = animation.FuncAnimation(fig, update, frames=frames_overlay, interval=1000/fps, blit=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"rollout_step{current_step}-{timestamp}"
            ani_save_path = ani_save_dir / f"{base_name}.mp4"
            ani.save(ani_save_path, writer="ffmpeg", fps=fps)
            plt.close(fig)

            # Torque plotting
            if len(torque_history) > 0:
                try:
                    torque_arr = np.stack(torque_history, axis=0)  # (T, 10)
                    T = torque_arr.shape[0]
                    # Use control step for horizontal axis to reflect real-time
                    time_axis = np.arange(T) * dt_frame

                    n_joints = torque_arr.shape[1]
                    rows, cols = 5, 2
                    fig_t, axes_t = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
                    axes_ft = axes_t.flatten()

                    joint_names = [
                        "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                        "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
                    ]

                    # Left leg torques
                    for row, j in enumerate(range(0, 5)):
                        axt_left = axes_ft[row * cols]
                        axt_left.plot(time_axis, torque_arr[:, j], label='Torque', color='tab:purple', linewidth=1.2)
                        axt_left.set_title(f"{joint_names[j]}")
                        axt_left.set_ylabel('Torque (Nm)')
                        axt_left.grid(alpha=0.3)

                    # Right leg torques
                    for row, j in enumerate(range(5, 10)):
                        axt_right = axes_ft[row * cols + 1]
                        axt_right.plot(time_axis, torque_arr[:, j], label='Torque', color='tab:purple', linewidth=1.2)
                        axt_right.set_title(f"{joint_names[j]}")
                        axt_right.grid(alpha=0.3)

                    # X labels bottom row
                    axes_ft[8].set_xlabel('Time (s)')
                    axes_ft[9].set_xlabel('Time (s)')
                    fig_t.suptitle('Applied PD Torques per Joint')
                    fig_t.tight_layout(rect=[0, 0, 1, 0.96])

                    torque_plot_path = ani_save_dir / f"{base_name}_torques.png"
                    fig_t.savefig(torque_plot_path, dpi=160)
                    plt.close(fig_t)
                    logging.info(f"Saved torque montage to {torque_plot_path}")
                except Exception as e:
                    logging.exception(f"Failed generating/saving torque montage: {e}")
            else:
                logging.warning("No torque data collected; episode ended before first step?")

            # Joint angle delta plotting (from observations)
            if len(angle_delta_history) > 0:
                try:
                    angle_arr = np.stack(angle_delta_history, axis=0)  # (T, 10)
                    T_angles = angle_arr.shape[0]
                    time_axis_angles = np.arange(T_angles) * dt_frame

                    n_joints = angle_arr.shape[1]
                    rows, cols = 5, 2
                    fig_a, axes_a = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
                    axes_fa = axes_a.flatten()

                    joint_names = [
                        "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                        "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
                    ]

                    # Left leg angle deltas
                    for row, j in enumerate(range(0, 5)):
                        ax_left = axes_fa[row * cols]
                        ax_left.plot(time_axis_angles, angle_arr[:, j], label='Angle Δ', color='tab:blue', linewidth=1.2)
                        ax_left.set_title(f"{joint_names[j]}")
                        ax_left.set_ylabel('Δ Angle (rad)')
                        ax_left.grid(alpha=0.3)

                    # Right leg angle deltas
                    for row, j in enumerate(range(5, 10)):
                        ax_right = axes_fa[row * cols + 1]
                        ax_right.plot(time_axis_angles, angle_arr[:, j], label='Angle Δ', color='tab:blue', linewidth=1.2)
                        ax_right.set_title(f"{joint_names[j]}")
                        ax_right.grid(alpha=0.3)

                    # X labels bottom row
                    axes_fa[8].set_xlabel('Time (s)')
                    axes_fa[9].set_xlabel('Time (s)')
                    fig_a.suptitle('Joint Angle Deltas (from observations)')
                    fig_a.tight_layout(rect=[0, 0, 1, 0.96])

                    angles_plot_path = ani_save_dir / f"{base_name}_angle_deltas.png"
                    fig_a.savefig(angles_plot_path, dpi=160)
                    plt.close(fig_a)
                    logging.info(f"Saved joint angle delta montage to {angles_plot_path}")
                except Exception as e:
                    logging.exception(f"Failed generating/saving joint angle delta montage: {e}")
            else:
                logging.warning("No joint angle delta data collected; episode ended before first step?")

            end_time = time.time()
            duration = timedelta(seconds=end_time - start_time)
            logging.info(f"Saved rollout video to {ani_save_path}")
            logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logging.info(f"Duration of visualization update: {duration}")
            logging.info("----------------")

        except Exception as e:
            logging.exception(f"Failed to generate/save rollout video at step {current_step}: {e}")

