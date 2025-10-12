"""Callback functions passed into PPO training loop."""
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from absl import logging
from datetime import datetime, timedelta
from pathlib import Path
import time

import matplotlib.pyplot as plt
import mujoco as mj
import jax
import numpy as np
import cv2
from brax.training import types
import io
import contextlib

from cassie_arl.config.cassie_consts import (
    FOOT_OFFSET,
    QPosIdx,
    JntRangeIdx
)


logging.set_verbosity(logging.INFO)


class ProgressCallback:
    """Callable progress callback for Brax PPO training loop."""
    def __init__(
            self,
            num_timesteps: dict,
            script_dir: Path,
            train_id: str,
            iteration: int,
            save_plot: bool = True
        ):
        self.num_timesteps = num_timesteps
        self.script_dir = script_dir
        self.iteration = iteration
        self.save_plot = save_plot
        self.save_dir = self.script_dir / "progress" / train_id
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
        plt.xlim([0, self.num_timesteps * 1.2])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={self.y_data[-1]:.3f}")
        
        if not self.save_plot:
            plt.pause(0.005)  # Small pause to update the figure
            return

        save_path = self.save_dir / f"progress_{self.iteration:02d}.png"
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
        actions_history: List[np.ndarray] = field(default_factory=list)  # (T, 10) first 10 action dims (normalized joint deltas)
        motor_qpos_history: List[np.ndarray] = field(default_factory=list)  # (T, 10) actual joint angles AFTER step
        target_history: List[np.ndarray] = field(default_factory=list)  # (T, 10) PD position targets used for control
        p_gain_history: List[np.ndarray] = field(default_factory=list)  # (T, 3) grouped Kp values (scaled)
        d_gain_history: List[np.ndarray] = field(default_factory=list)  # (T, 3) grouped Kd values (scaled)
        observation_history: List[np.ndarray] = field(default_factory=list)  # (T, obs_dim) most recent single-step obs
        cumulative_reward: float = 0.0

    # ------------------------------- Init ----------------------------------------
    def __init__(
            self,
            env,
            jit_reset,
            jit_step,
            script_dir: Path,
            train_id: str,
            iteration: int,
            run_every_n_calls: int = 1,
            skip_first_n_calls: int = 0,
            test_mode=False,
            video_fps: int = 30,
            video_scale: float = 1.0,
    ):
        self.env = env
        self.jit_reset = jit_reset
        self.jit_step = jit_step
        self.script_dir = script_dir
        self.run_every_n_calls = run_every_n_calls
        self.skip_first_n_calls = skip_first_n_calls
        self.test_mode = test_mode  # If True, save animation and plots to a test directory
        self._call_count = 0

        # Video settings
        self.video_fps = video_fps
        self.video_scale = video_scale

        # Centralized directory for all visualization artifacts
        # Keeping the same path as before for backward compatibility
        self.ani_save_dir = self.script_dir / "simulation" / train_id / f"iter_{iteration:02d}"
        if self.test_mode:
            self.ani_save_dir = self.ani_save_dir / "test"
        self.ani_save_dir.mkdir(parents=True, exist_ok=True)

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
        if self._call_count <= self.skip_first_n_calls:
            logging.info(f"Skipping visualization call {self._call_count} (first {self.skip_first_n_calls})")
            return False
        return True

    # ----------------------------- Logging Helpers -------------------------------
    def _log_header(self, current_step: int):
        print("")
        logging.info(f"--- Visualization update {self._call_count} ---")
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
            action, _ = inference_fn(state.obs, act_rng)
            # Let the environment perform its own PD; pass raw action
            state = self.jit_step(state, action)

            if bool(state.done):
                break

            # --- Collect per-step diagnostics ---
            data.traj.append(state)
            # Actions (first 10 dims are normalized joint deltas)
            try:
                nu = int(self.env.mjx_model.nu)
                data.actions_history.append(np.array(action[:nu]))
                # Record learned grouped PD gains (scaled to actual Kp/Kd)
                p_raw = np.array(action[nu:nu+3])
                d_raw = np.array(action[nu+3:nu+6])
                p_scaled = (p_raw + 1.0) / 2.0 * float(self.env._config.max_p_gain)
                d_scaled = (d_raw + 1.0) / 2.0 * float(self.env._config.max_d_gain)
                data.p_gain_history.append(p_scaled)
                data.d_gain_history.append(d_scaled)
            except Exception:
                pass
            # Torques are included in the latest observation (last 10 dims of single obs)
            try:
                latest_obs = np.array(state.info["obs_history"][0])
                nu = int(self.env.mjx_model.nu)
                data.torque_history.append(latest_obs[-nu:])
                # Record the most recent single-step observation for overlays
                data.observation_history.append(latest_obs)
            except Exception:
                pass
            # Store targets used for this control step (raw + filtered) and resulting joint angles AFTER physics step
            motor_qpos_new = state.data.qpos[QPosIdx.MOTORS]
            data.motor_qpos_history.append(np.array(motor_qpos_new))
            if "pos_targets" in state.info:
                data.target_history.append(np.array(state.info["pos_targets"]))
            self._record_reward(state, data)
            self._record_feet(state, data)
        return data

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
        # Subsample frames to target FPS to cut encoding time and memory
        ctrl_dt = float(self.env._config.ctrl_dt)
        fps_in = max(1.0 / ctrl_dt, 1.0)
        stride = max(1, int(round(fps_in / float(self.video_fps))))
        idxs = list(range(0, len(rollout.traj), stride)) if len(rollout.traj) > 0 else []
        # Effective dt between video frames
        dt_video = ctrl_dt * stride
        # Stream render + overlay + encode to avoid holding all frames in memory
        base_name, ani_save_path, _ = self._save_video_streaming(current_step, rollout, idxs, dt_video)
        # Plots use simulation dt (not video dt)
        self._plot_reward_components(rollout.reward_records, base_name, ctrl_dt)
        self._plot_torques(rollout.torque_history, base_name, ctrl_dt)
        self._plot_actions(rollout.actions_history, base_name, ctrl_dt)
        self._plot_gains(rollout.p_gain_history, rollout.d_gain_history, base_name, ctrl_dt)
        self._plot_joint_angles(
            rollout.motor_qpos_history,
            rollout.target_history,
            base_name,
            ctrl_dt,
        )
        if ani_save_path is not None:
            logging.info(f"Saved rollout video to {ani_save_path}")

    def _get_observation_overlay_text(self, obs_data, foot):
        """Prepare left-column observation text using latest single-step observation.

        Observation layout (current CassieEnv):
          [0]     pelvis_height (1)
          [1:5]   tilt_quat (4)
          [5:15]  motor_qpos_delta (10)
          [15:18] lin_vel_body (3)
          [18:21] ang_vel_body (3)
          [21:31] motor_qvel (10)
          [31:33] feet_contact (2)
          [33:43] last_torques (10)
        """
        if obs_data is None:
            return ["Observations: Not available"]

        lines = []
        lines.append("=== OBSERVATIONS ===")
        try:
            # Pelvis height
            lines.append(f"Pelvis height: {obs_data[0]:.3f} m")

            # Tilt quaternion and derived RPY
            qw, qx, qy, qz = obs_data[1:5]
            lines.append(f"Tilt quat: [{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}]")
            import math
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            lines.append(f"Tilt RPY (deg): [{math.degrees(roll):.1f}, {math.degrees(pitch):.1f}, {math.degrees(yaw):.1f}]")

            # Body-frame velocities
            lin_vel = obs_data[15:18]
            ang_vel = obs_data[18:21]
            lines.append(f"Lin vel [x,y,z]: [{lin_vel[0]:.2f}, {lin_vel[1]:.2f}, {lin_vel[2]:.2f}]")
            lines.append(f"Ang vel [x,y,z]: [{ang_vel[0]:.2f}, {ang_vel[1]:.2f}, {ang_vel[2]:.2f}]")

            # Foot heights
            lines.append("--- Foot Heights ---")
            lines.append(f"L foot z: {foot.left_foot_z:.3f} m")
            lines.append(f"R foot z: {foot.right_foot_z:.3f} m")
            # Foot contacts from observation (indices 31:33)
            try:
                fc = obs_data[31:33]
                l_contact = bool(fc[0] >= 0.5)
                r_contact = bool(fc[1] >= 0.5)
                lines.append(f"Contacts: L={l_contact}  R={r_contact}")
            except Exception:
                pass
        except Exception:
            # Fallback minimal info if parsing fails
            lines = ["Observations: parse error"]
        return lines

    def _get_reward_overlay_text(self, reward_record, reward_scales: Dict[str, float]):
        """Prepare right-column reward text showing raw per-step components with actual signs (exclude 'alive')."""
        lines = [
            f"Step Reward: {reward_record.step_reward:.4f}",
            f"Cumulative: {reward_record.cumulative_reward:.4f}",
            "--- Rewards ---",
        ]
        for name, weighted_value in reward_record.components.items():
            scale = reward_scales.get(name, 1.0)
            raw_value = weighted_value / scale if scale != 0 else 0.0
            lines.append(f"{name}: {raw_value * scale:.4f}")
        return lines

    # ----------------------------- Saving Artifacts ------------------------------
    def _save_video_streaming(
            self,
            current_step: int,
            rollout: 'VisualizePolicyCallback.RolloutData',
            idxs: List[int],
            dt_frame: float
    ) -> tuple[str, Path | None, float]:
        """Render, overlay, and encode video in a streaming fashion to avoid large memory spikes.

        Returns base_name, path, and control dt (for plots).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"rollout_step{current_step}-{timestamp}"
        ani_save_path = self.ani_save_dir / f"{base_name}.mp4"

        if len(idxs) == 0:
            logging.warning("No indices to render; skipping video.")
            return base_name, None, self.env._config.ctrl_dt

        # Pre-create scene option once
        scene_option = mj.MjvOption()
        scene_option.geomgroup[1] = True  # Show frames?
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_COM] = True
        scene_option.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = True

        # Helper to render a single frame for a given state index
        def render_one(i: int) -> np.ndarray:
            # env.render accepts a trajectory; pass single-element list and take first image
            # Silence tqdm or other progress logging inside render
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                fr_list = self.env.render([rollout.traj[i]], camera="track", scene_option=scene_option, width=640*2, height=480)
            return np.array(fr_list[0])

        # First frame to determine size and possibly scale policy
        fr0 = render_one(idxs[0])
        # Heuristic: auto downscale if resolution*frames is too large and user didn't set scale < 1
        auto_scale = 1.0
        try:
            h0, w0 = fr0.shape[:2]
            est_frames = len(idxs)
            est_megapixels = (h0 * w0 * est_frames) / 1e6
            if self.video_scale == 1.0 and est_megapixels > 400:  # ~400 MP threshold (~400MB of raw RGB)
                auto_scale = 0.5
                logging.info(f"Auto-downscaling video by 0.5x due to size: {w0}x{h0}x{est_frames}")
        except Exception:
            h0, w0 = fr0.shape[:2]

        scale = self.video_scale * auto_scale
        if abs(scale - 1.0) > 1e-6:
            nh = max(1, int(round(h0 * scale)))
            nw = max(1, int(round(w0 * scale)))
        else:
            nh, nw = h0, w0

        fps_out = float(1.0 / max(dt_frame, 1e-6))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # portable
        writer = cv2.VideoWriter(str(ani_save_path), fourcc, fps_out, (nw, nh))
        if not writer.isOpened():
            logging.error("Failed to open video writer")
            return base_name, None, self.env._config.ctrl_dt

        try:
            reward_scales = {k: float(v) for k, v in self.env._config.reward_config.weights.items()}
            for fi, i in enumerate(idxs):
                # Render
                fr = fr0 if (fi == 0) else render_one(i)
                # Ensure uint8 RGB
                if fr.dtype != np.uint8:
                    fr = (255 * np.clip(fr, 0, 1)).astype(np.uint8) if fr.dtype in (np.float32, np.float64) else fr.astype(np.uint8)

                # Overlay (in place)
                frame_rgb = fr.copy()
                # Build overlay text using on-the-fly access to rollout buffers
                try:
                    foot = rollout.foot_infos[i]
                except Exception:
                    foot = self.FootInfo(0.0, 0.0, 0.0, 0.0)
                try:
                    reward_record = rollout.reward_records[i]
                except Exception:
                    reward_record = self.RewardRecord({}, 0.0, 0.0)
                obs_data = rollout.observation_history[i] if i < len(rollout.observation_history) else None
                left_lines = self._get_observation_overlay_text(obs_data, foot)
                right_lines = self._get_reward_overlay_text(reward_record, reward_scales)

                font_scale = 0.8
                thickness = 2
                for li, line in enumerate(left_lines):
                    self._put_text(frame_rgb, line, (10, 30 + li * 30), color=(255, 255, 255), font_scale=font_scale, thickness=thickness)
                fw = frame_rgb.shape[1]
                right_x = fw - 390
                for ri, line in enumerate(right_lines):
                    if (ri < 3) or (':' not in line):
                        color = (255, 255, 255)
                    else:
                        try:
                            value = float(line.split(': ')[-1])
                            color = (0, 255, 0) if value > 0 else (255, 0, 0)
                        except Exception:
                            color = (255, 255, 255)
                    self._put_text(frame_rgb, line, (right_x, 30 + ri * 30), color=color, font_scale=font_scale, thickness=thickness)
                # Simulation time (bottom-left)
                time_text = f"t = {fi * dt_frame:.2f} s"
                fh = frame_rgb.shape[0]
                self._put_text(frame_rgb, time_text, (10, fh - 10), color=(255, 255, 255), font_scale=font_scale, thickness=thickness)

                # Optional scale
                if (nh, nw) != frame_rgb.shape[:2]:
                    frame_rgb = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

                # Convert to BGR for OpenCV and write
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
        except Exception as e:
            logging.exception(f"Failed during streaming video encoding: {e}")
            try:
                writer.release()
            except Exception:
                pass
            return base_name, None, self.env._config.ctrl_dt
        finally:
            try:
                writer.release()
            except Exception:
                pass

        logging.info(f"Saved rollout video to {ani_save_path}")
        return base_name, ani_save_path, self.env._config.ctrl_dt

    def _put_text(self, img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], font_scale: float = 0.8, thickness: int = 2):
        """Draw readable text: solid dark background + outline to reduce blur.

        Note: img is RGB; OpenCV writes BGR values directly into the array channels,
        but since we display with Matplotlib (RGB), providing standard tuples works
        as expected for white/green/red.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = org
        # Background rectangle (solid) to maximize contrast
        x0 = max(x - 4, 0)
        y0 = max(y - th - 6, 0)
        x1 = min(x + tw + 4, img.shape[1] - 1)
        y1 = min(y + baseline + 4, img.shape[0] - 1)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        # Outline (black, thicker)
        cv2.putText(img, text, org, font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_8)
        # Foreground text
        cv2.putText(img, text, org, font, font_scale, color, thickness, lineType=cv2.LINE_8)

    def _plot_torques(self, torque_history: List[np.ndarray], base_name: str, dt_frame: float):
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
            torque_lowers = np.array(self.env._torque_lowers)
            torque_uppers = np.array(self.env._torque_uppers)
            for row, j in enumerate(range(0, 5)):
                axt_left = axes_ft[row * cols]
                axt_left.plot(time_axis, torque_arr[:, j], color='tab:purple', linewidth=1.2)

                # Add horizontal lines for target limits
                axt_left.axhline(torque_lowers[j], color='tab:red', linestyle='--', linewidth=1, label='Torque lower limit' if row == 0 else "")
                axt_left.axhline(torque_uppers[j], color='tab:red', linestyle='--', linewidth=1, label='Torque upper limit' if row == 0 else "")

                axt_left.set_title(joint_names[j])
                axt_left.set_ylabel('Torque (Nm)')
                axt_left.grid(alpha=0.3)
            for row, j in enumerate(range(5, 10)):
                axt_right = axes_ft[row * cols + 1]
                axt_right.plot(time_axis, torque_arr[:, j], color='tab:purple', linewidth=1.2)

                # Add horizontal lines for target limits
                axt_right.axhline(torque_lowers[j], color='tab:red', linestyle='--', linewidth=1, label='Torque lower limit' if row == 0 else "")
                axt_right.axhline(torque_uppers[j], color='tab:red', linestyle='--', linewidth=1, label='Torque upper limit' if row == 0 else "")

                axt_right.set_title(joint_names[j])
                axt_right.grid(alpha=0.3)
            axes_ft[8].set_xlabel('Time (s)')
            axes_ft[9].set_xlabel('Time (s)')
            fig_t.suptitle('Applied PD Torques per Joint')
            fig_t.tight_layout(rect=[0, 0, 1, 0.96])
            torque_plot_path = self.ani_save_dir / f"{base_name}_torques.png"
            fig_t.savefig(torque_plot_path, dpi=160)
            plt.close(fig_t)
            logging.info(f"Saved torque montage to {torque_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving torque montage: {e}")

    def _plot_actions(self, actions_history: List[np.ndarray], base_name: str, dt_frame: float):
        if len(actions_history) == 0:
            logging.warning("No action data collected; episode ended before first step?")
            return
        try:
            act_arr = np.stack(actions_history, axis=0)
            time_axis = np.arange(act_arr.shape[0]) * dt_frame
            rows, cols = 5, 2
            fig_a, axes_a = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
            axes_fa = axes_a.flatten()
            joint_names = [
                "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
            ]
            for row, j in enumerate(range(0, 5)):
                ax_left = axes_fa[row * cols]
                ax_left.plot(time_axis, act_arr[:, j], color='tab:orange', linewidth=1.2)
                ax_left.set_title(joint_names[j])
                ax_left.set_ylabel('Action (norm)')
                ax_left.set_ylim([-1.05, 1.05])
                ax_left.grid(alpha=0.3)
            for row, j in enumerate(range(5, 10)):
                ax_right = axes_fa[row * cols + 1]
                ax_right.plot(time_axis, act_arr[:, j], color='tab:orange', linewidth=1.2)
                ax_right.set_title(joint_names[j])
                ax_right.set_ylim([-1.05, 1.05])
                ax_right.grid(alpha=0.3)
            axes_fa[8].set_xlabel('Time (s)')
            axes_fa[9].set_xlabel('Time (s)')
            fig_a.suptitle('Normalized Joint Actions')
            fig_a.tight_layout(rect=[0, 0, 1, 0.96])
            actions_plot_path = self.ani_save_dir / f"{base_name}_actions.png"
            fig_a.savefig(actions_plot_path, dpi=160)
            plt.close(fig_a)
            logging.info(f"Saved actions montage to {actions_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving actions montage: {e}")

    def _plot_gains(
        self,
        p_gain_history: List[np.ndarray],
        d_gain_history: List[np.ndarray],
        base_name: str,
        dt_frame: float,
    ):
        if len(p_gain_history) == 0 or len(d_gain_history) == 0:
            logging.warning("No PD gain data collected; episode ended before first step?")
            return
        try:
            p_arr = np.stack(p_gain_history, axis=0)  # (T, 3)
            d_arr = np.stack(d_gain_history, axis=0)  # (T, 3)
            time_axis = np.arange(p_arr.shape[0]) * dt_frame
            fig, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
            labels = ["Hip roll/yaw", "Hip pitch/knee", "Foot"]
            colors = ['tab:blue', 'tab:green', 'tab:red']
            # Plot Kp
            for i in range(3):
                ax_p.plot(time_axis, p_arr[:, i], label=labels[i], color=colors[i], linewidth=1.5)
            ax_p.set_title('Learned Kp (grouped)')
            ax_p.set_ylabel('Kp')
            ax_p.set_xlabel('Time (s)')
            ax_p.grid(alpha=0.3)
            ax_p.set_ylim([0, float(self.env._config.max_p_gain) * 1.05])
            ax_p.legend(frameon=False, fontsize=8)
            # Plot Kd
            for i in range(3):
                ax_d.plot(time_axis, d_arr[:, i], label=labels[i], color=colors[i], linewidth=1.5)
            ax_d.set_title('Learned Kd (grouped)')
            ax_d.set_ylabel('Kd')
            ax_d.set_xlabel('Time (s)')
            ax_d.grid(alpha=0.3)
            ax_d.set_ylim([0, float(self.env._config.max_d_gain) * 1.05])
            # Only one legend is needed; keep it on Kp subplot
            fig.suptitle('Learned PD Gains Over Time')
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            gains_plot_path = self.ani_save_dir / f"{base_name}_pd_gains.png"
            fig.savefig(gains_plot_path, dpi=160)
            plt.close(fig)
            logging.info(f"Saved PD gains figure to {gains_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving PD gains figure: {e}")

    def _plot_reward_components(
        self,
        reward_records: List['VisualizePolicyCallback.RewardRecord'],
        base_name: str,
        dt_frame: float,
    ):
        """Plot per-step weighted reward components over time (excluding 'alive')."""
        if len(reward_records) == 0:
            logging.warning("No reward data collected; skipping reward component plot.")
            return
        try:
            # Union of component names across steps, excluding 'alive'
            all_keys: List[str] = []
            seen = set()
            for rr in reward_records:
                for k in rr.components.keys():
                    if k == 'alive':
                        continue
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            if len(all_keys) == 0:
                logging.warning("Reward records contain no component entries (excluding 'alive'); skipping plot.")
                return
            T = len(reward_records)
            comp_matrix = np.zeros((T, len(all_keys)), dtype=float)
            for t, rr in enumerate(reward_records):
                for k_i, k in enumerate(all_keys):
                    if k in rr.components:
                        comp_matrix[t, k_i] = rr.components[k]
            time_axis = np.arange(T) * dt_frame
            fig_rc, ax_rc = plt.subplots(figsize=(10, 6))
            for k_i, k in enumerate(all_keys):
                ax_rc.plot(time_axis, comp_matrix[:, k_i], label=k, linewidth=1.3)
            ax_rc.set_xlabel('Time (s)')
            ax_rc.set_ylabel('Weighted Component Value')
            ax_rc.set_title('Reward Components (per step, weighted, excl. alive)')
            ax_rc.grid(alpha=0.3)
            ax_rc.legend(frameon=False, fontsize=8, ncol=2 if len(all_keys) > 8 else 1)
            fig_rc.tight_layout()
            reward_plot_path = self.ani_save_dir / f"{base_name}_reward_components.png"
            fig_rc.savefig(reward_plot_path, dpi=160)
            plt.close(fig_rc)
            logging.info(f"Saved reward components plot to {reward_plot_path}")
        except Exception as e:
            logging.exception(f"Failed generating/saving reward components plot: {e}")

    def _plot_joint_angles(
        self,
        motor_qpos_history: List[np.ndarray],
        target_history: List[np.ndarray],
        base_name: str,
        dt_frame: float,
    ):
        """Plot actual joint angles, PD targets, joint limits, and standing (neutral) pose."""
        if len(motor_qpos_history) == 0:
            logging.warning("No joint angle data collected; episode ended before first step?")
            return
        try:
            actual_arr = np.stack(motor_qpos_history, axis=0)  # (T, 10)
            targ_arr = np.stack(target_history, axis=0) if len(target_history) == len(motor_qpos_history) else None
            T = actual_arr.shape[0]
            time_axis = np.arange(T) * dt_frame

            rows, cols = 5, 2
            fig_j, axes_j = plt.subplots(rows, cols, figsize=(cols * 6, rows * 3), sharex=True)
            axes_fj = axes_j.flatten()

            joint_names = [
                "Left Hip Roll", "Left Hip Yaw", "Left Hip Pitch", "Left Knee", "Left Foot",
                "Right Hip Roll", "Right Hip Yaw", "Right Hip Pitch", "Right Knee", "Right Foot"
            ]

            # Get joint ranges and neutral standing pose
            jnt_lowers = np.array(self.env._mj_model.jnt_range[JntRangeIdx.MOTORS, 0])
            jnt_uppers = np.array(self.env._mj_model.jnt_range[JntRangeIdx.MOTORS, 1])
            jnt_neutral = np.array(self.env._standing_jnt_angles)

            target_jnt_lowers = np.array(self.env._standing_jnt_angles - self.env._max_jnt_deltas)
            target_jnt_uppers = np.array(self.env._standing_jnt_angles + self.env._max_jnt_deltas)

            # Left side joints (indices 0..4) -> left column
            for row, j in enumerate(range(0, 5)):
                ax_left = axes_fj[row * cols]
                ax_left.plot(time_axis, actual_arr[:, j], color='tab:blue', linewidth=1.4, label='Actual')
                if targ_arr is not None:
                    ax_left.plot(time_axis, targ_arr[:, j], color='tab:green', linewidth=1.2, linestyle='-', label='Target')

                # Add horizontal lines for limits and standing pose
                ax_left.axhline(jnt_lowers[j], color='tab:red', linestyle='--', linewidth=1, label='Lower limit' if row == 0 else "")
                ax_left.axhline(jnt_uppers[j], color='tab:red', linestyle='--', linewidth=1, label='Upper limit' if row == 0 else "")
                ax_left.axhline(jnt_neutral[j], color='tab:gray', linestyle=':', linewidth=1.2, label='Standing pose' if row == 0 else "")

                # Add horizontal lines for target limits
                ax_left.axhline(target_jnt_lowers[j], color='tab:purple', linestyle='--', linewidth=1, label='Target lower limit' if row == 0 else "")
                ax_left.axhline(target_jnt_uppers[j], color='tab:purple', linestyle='--', linewidth=1, label='Target upper limit' if row == 0 else "")

                ax_left.set_title(joint_names[j])
                ax_left.set_ylabel('Angle (rad)')
                ax_left.grid(alpha=0.3)

            # Right side joints (indices 5..9) -> right column
            for row, j in enumerate(range(5, 10)):
                ax_right = axes_fj[row * cols + 1]
                ax_right.plot(time_axis, actual_arr[:, j], color='tab:blue', linewidth=1.4, label='Actual')
                if targ_arr is not None:
                    ax_right.plot(time_axis, targ_arr[:, j], color='tab:green', linewidth=1.2, linestyle='-', label='Target')

                # Add horizontal lines for limits and standing pose
                ax_right.axhline(jnt_lowers[j], color='tab:red', linestyle='--', linewidth=1)
                ax_right.axhline(jnt_uppers[j], color='tab:red', linestyle='--', linewidth=1)
                ax_right.axhline(jnt_neutral[j], color='tab:gray', linestyle=':', linewidth=1.2)

                # Add horizontal lines for target limits
                ax_right.axhline(target_jnt_lowers[j], color='tab:purple', linestyle='--', linewidth=1)
                ax_right.axhline(target_jnt_uppers[j], color='tab:purple', linestyle='--', linewidth=1)

                ax_right.set_title(joint_names[j])
                ax_right.grid(alpha=0.3)

            # Common legend on the first subplot
            axes_fj[0].legend(frameon=False, fontsize=8, ncol=2, loc='upper right')

            axes_fj[8].set_xlabel('Time (s)')
            axes_fj[9].set_xlabel('Time (s)')

            fig_j.suptitle('Joint Angles: Actual vs PD Targets (with Limits & Standing Pose)')
            fig_j.tight_layout(rect=[0, 0, 1, 0.96])

            joint_plot_path = self.ani_save_dir / f"{base_name}_joint_angles.png"
            fig_j.savefig(joint_plot_path, dpi=160)
            plt.close(fig_j)
            logging.info(f"Saved joint angle montage to {joint_plot_path}")

        except Exception as e:
            logging.exception(f"Failed generating/saving joint angle montage: {e}")
