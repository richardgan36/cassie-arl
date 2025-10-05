"""Callback functions passed into PPO training loop."""
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from absl import logging
from datetime import datetime, timedelta
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mujoco as mj
import jax
import numpy as np
import jax.numpy as jnp
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
    """Callable visualization callback for Brax PPO training loop.

    The original monolithic implementation has been decomposed into modular
    helper methods so that functionality (data collection, rendering,
    overlays, artifact saving, plotting) can be added/removed independently.

    Public interface and produced artifacts are preserved.
    """

    # ------------------------------- Data Models ---------------------------------
    @dataclass
    class RewardRecord:
        components: Dict[str, float]
        step_reward: float
        cumulative_reward: float

    @dataclass
    class FootInfo:
        left_foot_z: float
        right_foot_z: float
        left_tarsus_z: float
        right_tarsus_z: float

    @dataclass
    class RolloutData:
        traj: List[Any] = field(default_factory=list)  # List of env states
        reward_records: List['VisualizePolicyCallback.RewardRecord'] = field(default_factory=list)
        foot_infos: List['VisualizePolicyCallback.FootInfo'] = field(default_factory=list)
        torque_history: List[np.ndarray] = field(default_factory=list)  # (T, 10)
        angle_delta_history: List[np.ndarray] = field(default_factory=list)  # (T, 10)
        raw_qvel_history: List[np.ndarray] = field(default_factory=list)  # (T, 10) raw motor velocities
        filt_qvel_history: List[np.ndarray] = field(default_factory=list)  # (T, 10) filtered motor velocities
        cumulative_reward: float = 0.0

    # ------------------------------- Init ----------------------------------------
    def __init__(self, env, jit_reset, jit_step, script_dir: Path, train_id: str, run_every_n_calls: int = 1):
        self.env = env
        self.jit_reset = jit_reset
        self.jit_step = jit_step
        self.script_dir = script_dir
        self.train_id = train_id
        self.run_every_n_calls = run_every_n_calls  # frequency control
        self._call_count = 0

        # Extension points (ordered). Each receives (current_step, rollout_data, context_dict)
        self.post_processors: List[Callable[[int, 'VisualizePolicyCallback.RolloutData', Dict[str, Any]], None]] = [
            self._render_and_save_artifacts
        ]

    # ------------------------------- Public API ----------------------------------
    def __call__(self, current_step: int, make_policy, params):
        """Entry point invoked by training loop."""
        try:
            if not self._should_run():
                return
            start_time = time.time()
            self._log_header(current_step)
            inference_fn = make_policy(params, deterministic=True)
            rollout = self._collect_rollout(current_step, inference_fn)
            if len(rollout.traj) == 0:
                logging.warning("No frames collected; skipping.")
                return

            context: Dict[str, Any] = {"start_time": start_time}
            for processor in self.post_processors:
                try:
                    processor(current_step, rollout, context)
                except Exception as e:  # Keep other processors running
                    logging.exception(f"Post-processor {processor.__name__} failed: {e}")
            self._log_footer(start_time)
        except Exception as e:
            logging.exception(f"Failed to generate/save rollout video at step {current_step}: {e}")

    # ----------------------------- Control Helpers -------------------------------
    def _should_run(self) -> bool:
        self._call_count += 1
        if (self._call_count % self.run_every_n_calls) != 0:
            logging.info(f"Skipping visualization call {self._call_count} (every {self.run_every_n_calls})")
            return False
        return True

    # ----------------------------- Logging Helpers -------------------------------
    def _log_header(self, current_step: int):
        print("")
        logging.info("--- Visualization update ---")
        logging.info(f"Generating rollout video at step {current_step}")

    def _log_footer(self, start_time: float):
        end_time = time.time()
        duration = timedelta(seconds=end_time - start_time)
        logging.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Duration of visualization update: {duration}")
        logging.info("----------------")

    # ----------------------------- Data Conversion -------------------------------
    def _reward_components_to_dict(self, rc) -> Dict[str, float]:
        """Best-effort conversion of reward components object to dict of floats."""
        try:
            if dataclasses.is_dataclass(rc):
                out: Dict[str, float] = {}
                for f in dataclasses.fields(rc):
                    out[f.name] = float(np.array(getattr(rc, f.name)))
                return out
            if isinstance(rc, dict):
                return {k: float(np.array(v)) for k, v in rc.items()}
        except Exception:
            pass
        return {}

    # ----------------------------- Rollout Collection ----------------------------
    def _collect_rollout(self, current_step: int, inference_fn) -> 'VisualizePolicyCallback.RolloutData':
        rng = jax.random.PRNGKey(int(current_step + 1) & 0xFFFFFFFF)
        state = self.jit_reset(rng)
        data = self.RolloutData()

        for _ in range(self.env._config.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = inference_fn(state.obs, act_rng)

            # --- Reproduce env.step path (custom PD) ---
            raw_pos_targets = self.env._action_to_jnt_targets(ctrl)
            target_filter = state.info["target_filter"]
            filtered_targets, new_target_filter = target_filter.apply(raw_pos_targets)
            # Re-implement PD control locally so we can capture raw + filtered velocities
            motor_qpos = state.data.qpos[QPosIdx.MOTORS]
            raw_motor_qvel = state.data.qvel[QVelIdx.MOTORS]
            vel_filter_state = state.info["vel_filter"]
            filt_vel, new_vel_filter = vel_filter_state.apply(raw_motor_qvel)
            pos_error = filtered_targets - motor_qpos
            vel_error = -filt_vel
            torque = self.env._p_gain * pos_error + self.env._d_gain * vel_error + self.env._standing_torques
            torque = self.env._limit_torque_rate(torque, state.info["last_torques"])  # type: ignore
            torque = jnp.clip(torque, self.env._torque_lowers, self.env._torque_uppers)  # type: ignore

            state = self.jit_step(state, torque)
            state.info["last_action"] = ctrl
            state.info["last_torques"] = torque
            state.info["vel_filter"] = new_vel_filter
            state.info["target_filter"] = new_target_filter

            if bool(state.done):
                break

            # --- Collect per-step diagnostics ---
            data.traj.append(state)
            data.torque_history.append(np.array(torque))
            data.raw_qvel_history.append(np.array(raw_motor_qvel))
            data.filt_qvel_history.append(np.array(filt_vel))
            self._maybe_record_angle_deltas(state, data)
            self._record_reward(state, data)
            self._record_feet(state, data)
        return data

    def _maybe_record_angle_deltas(self, state, data: 'VisualizePolicyCallback.RolloutData'):
        try:
            motor_delta = np.array(state.obs[5:15])
            if motor_delta.shape[0] == 10:
                data.angle_delta_history.append(motor_delta)
        except Exception:
            pass

    def _record_reward(self, state, data: 'VisualizePolicyCallback.RolloutData'):
        step_reward = float(state.reward)
        data.cumulative_reward += step_reward
        components = self._reward_components_to_dict(state.info.get("reward_components", {}))
        data.reward_records.append(self.RewardRecord(components, step_reward, data.cumulative_reward))

    def _record_feet(self, state, data: 'VisualizePolicyCallback.RolloutData'):
        left_foot_z = float(np.array(state.data.xpos[self.env._left_foot_id, 2])) - FOOT_OFFSET
        right_foot_z = float(np.array(state.data.xpos[self.env._right_foot_id, 2])) - FOOT_OFFSET
        fi = self.FootInfo(
            left_foot_z=left_foot_z,
            right_foot_z=right_foot_z,
            left_tarsus_z=float(np.array(state.data.xpos[self.env._left_tarsus_id, 2])),
            right_tarsus_z=float(np.array(state.data.xpos[self.env._right_tarsus_id, 2])),
        )
        data.foot_infos.append(fi)

    # ----------------------------- Rendering & Overlays --------------------------
    def _render_and_save_artifacts(self, current_step: int, rollout: 'VisualizePolicyCallback.RolloutData', context: Dict[str, Any]):
        frames = self._render_frames(rollout.traj)
        frames_overlay = self._apply_overlays(frames, rollout)
        base_name, ani_save_dir, ani_save_path, dt_frame = self._save_video(current_step, frames_overlay)
        # Plots
        self._plot_torques(rollout.torque_history, base_name, ani_save_dir, dt_frame)
        self._plot_angle_deltas(rollout.angle_delta_history, base_name, ani_save_dir, dt_frame)
        self._plot_qvels(rollout.raw_qvel_history, rollout.filt_qvel_history, base_name, ani_save_dir, dt_frame)
        logging.info(f"Saved rollout video to {ani_save_path}")

    def _render_frames(self, traj: List[Any]):
        scene_option = mj.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = False
        return self.env.render(traj, camera="track", scene_option=scene_option, width=640*2, height=480)

    def _apply_overlays(self, frames, rollout: 'VisualizePolicyCallback.RolloutData'):
        reward_scales = {k: float(v) for k, v in self.env._config.reward_config.scales.items()}
        out_frames = []
        for idx, frame in enumerate(frames):
            frame_rgb = np.array(frame).copy()
            foot = rollout.foot_infos[idx]
            reward_record = rollout.reward_records[idx]
            # Left column
            left_lines = [
                f"Left foot z: {foot.left_foot_z:.3f} m",
                f"Right foot z: {foot.right_foot_z:.3f} m",
                f"Left tarsus z: {foot.left_tarsus_z:.3f} m",
                f"Right tarsus z: {foot.right_tarsus_z:.3f} m",
            ]
            # Right column header
            right_lines = [
                f"Step Reward: {reward_record.step_reward:.4f}",
                f"Cumulative Reward: {reward_record.cumulative_reward:.4f}",
                "-----------------------------",
                "Component           Raw Value | Weighted",
            ]
            # Components with raw + weighted
            for name, weighted_value in reward_record.components.items():
                scale = reward_scales.get(name, 1.0)
                raw_value = weighted_value / scale if scale != 0 else 0.0
                name_pad = f"{name}:".ljust(18)
                right_lines.append(f"{name_pad} {raw_value:>8.4f} | {weighted_value:>8.4f}")

            # Draw text
            for li, line in enumerate(left_lines):
                cv2.putText(frame_rgb, line, (10, 30 + li * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fw = frame_rgb.shape[1]
            right_x = fw - 500
            for ri, line in enumerate(right_lines):
                color = (255, 255, 255)
                if ri >= 4:  # component rows
                    cname = line.split(':')[0]
                    if cname in reward_record.components:
                        val = reward_record.components[cname]
                        if val > 0:
                            color = (0, 255, 0)
                        elif val < 0:
                            color = (255, 100, 100)
                cv2.putText(frame_rgb, line, (right_x, 30 + ri * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out_frames.append(frame_rgb)
        return out_frames

    # ----------------------------- Saving Artifacts ------------------------------
    def _save_video(self, current_step: int, frames_overlay: List[np.ndarray]):
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

        ani = animation.FuncAnimation(fig, update, frames=frames_overlay, interval=1000 / fps, blit=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"rollout_step{current_step}-{timestamp}"
        ani_save_path = ani_save_dir / f"{base_name}.mp4"
        ani.save(ani_save_path, writer="ffmpeg", fps=fps)
        plt.close(fig)
        return base_name, ani_save_dir, ani_save_path, dt_frame

    def _plot_torques(self, torque_history: List[np.ndarray], base_name: str, save_dir: Path, dt_frame: float):
        if len(torque_history) == 0:
            logging.warning("No torque data collected; episode ended before first step?")
            return
        try:
            torque_arr = np.stack(torque_history, axis=0)
            time_axis = np.arange(torque_arr.shape[0]) * dt_frame
            rows, cols = 5, 2
            fig_t, axes_t = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
            axes_ft = axes_t.flatten()
            joint_names = [
                "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
            ]
            for row, j in enumerate(range(0, 5)):
                axt_left = axes_ft[row * cols]
                axt_left.plot(time_axis, torque_arr[:, j], color='tab:purple', linewidth=1.2)
                axt_left.set_title(joint_names[j])
                axt_left.set_ylabel('Torque (Nm)')
                axt_left.grid(alpha=0.3)
            for row, j in enumerate(range(5, 10)):
                axt_right = axes_ft[row * cols + 1]
                axt_right.plot(time_axis, torque_arr[:, j], color='tab:purple', linewidth=1.2)
                axt_right.set_title(joint_names[j])
                axt_right.grid(alpha=0.3)
            axes_ft[8].set_xlabel('Time (s)')
            axes_ft[9].set_xlabel('Time (s)')
            fig_t.suptitle('Applied PD Torques per Joint')
            fig_t.tight_layout(rect=[0, 0, 1, 0.96])
            torque_plot_path = save_dir / f"{base_name}_torques.png"
            fig_t.savefig(torque_plot_path, dpi=160)
            plt.close(fig_t)
            logging.info(f"Saved torque montage to {torque_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving torque montage: {e}")

    def _plot_angle_deltas(self, angle_delta_history: List[np.ndarray], base_name: str, save_dir: Path, dt_frame: float):
        if len(angle_delta_history) == 0:
            logging.warning("No joint angle delta data collected; episode ended before first step?")
            return
        try:
            angle_arr = np.stack(angle_delta_history, axis=0)
            time_axis = np.arange(angle_arr.shape[0]) * dt_frame
            rows, cols = 5, 2
            fig_a, axes_a = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
            axes_fa = axes_a.flatten()
            joint_names = [
                "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
            ]
            for row, j in enumerate(range(0, 5)):
                ax_left = axes_fa[row * cols]
                ax_left.plot(time_axis, angle_arr[:, j], color='tab:blue', linewidth=1.2)
                ax_left.set_title(joint_names[j])
                ax_left.set_ylabel('Δ Angle (rad)')
                ax_left.grid(alpha=0.3)
            for row, j in enumerate(range(5, 10)):
                ax_right = axes_fa[row * cols + 1]
                ax_right.plot(time_axis, angle_arr[:, j], color='tab:blue', linewidth=1.2)
                ax_right.set_title(joint_names[j])
                ax_right.grid(alpha=0.3)
            axes_fa[8].set_xlabel('Time (s)')
            axes_fa[9].set_xlabel('Time (s)')
            fig_a.suptitle('Joint Angle Deltas (from observations)')
            fig_a.tight_layout(rect=[0, 0, 1, 0.96])
            angles_plot_path = save_dir / f"{base_name}_angle_deltas.png"
            fig_a.savefig(angles_plot_path, dpi=160)
            plt.close(fig_a)
            logging.info(f"Saved joint angle delta montage to {angles_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving joint angle delta montage: {e}")

    def _plot_qvels(
        self,
        raw_qvel_history: List[np.ndarray],
        filt_qvel_history: List[np.ndarray],
        base_name: str,
        save_dir: Path,
        dt_frame: float,
    ):
        """Plot raw vs filtered joint velocities over the rollout."""
        if len(raw_qvel_history) == 0 or len(filt_qvel_history) == 0:
            logging.warning("No joint velocity data collected; episode ended before first step?")
            return
        try:
            raw_arr = np.stack(raw_qvel_history, axis=0)  # (T, 10)
            filt_arr = np.stack(filt_qvel_history, axis=0)  # (T, 10)
            T = raw_arr.shape[0]
            time_axis = np.arange(T) * dt_frame
            rows, cols = 5, 2
            fig_v, axes_v = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
            axes_fv = axes_v.flatten()
            joint_names = [
                "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
            ]
            for row, j in enumerate(range(0, 5)):
                ax_left = axes_fv[row * cols]
                ax_left.plot(time_axis, raw_arr[:, j], color='lightcoral', linewidth=0.9, label='Raw')
                ax_left.plot(time_axis, filt_arr[:, j], color='tab:green', linewidth=1.2, label='Filtered')
                ax_left.set_title(joint_names[j])
                ax_left.set_ylabel('qvel (rad/s)')
                ax_left.grid(alpha=0.3)
                if row == 0:
                    ax_left.legend(frameon=False, fontsize=8)
            for row, j in enumerate(range(5, 10)):
                ax_right = axes_fv[row * cols + 1]
                ax_right.plot(time_axis, raw_arr[:, j], color='lightcoral', linewidth=0.9, label='Raw')
                ax_right.plot(time_axis, filt_arr[:, j], color='tab:green', linewidth=1.2, label='Filtered')
                ax_right.set_title(joint_names[j])
                ax_right.grid(alpha=0.3)
            axes_fv[8].set_xlabel('Time (s)')
            axes_fv[9].set_xlabel('Time (s)')
            fig_v.suptitle('Joint Velocities (Raw vs Filtered)')
            fig_v.tight_layout(rect=[0, 0, 1, 0.96])
            qvel_plot_path = save_dir / f"{base_name}_qvels.png"
            fig_v.savefig(qvel_plot_path, dpi=160)
            plt.close(fig_v)
            logging.info(f"Saved joint velocity montage to {qvel_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving joint velocity montage: {e}")

