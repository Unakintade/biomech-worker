"""
Hybrid pipeline: MuJoCo geometric IK + inverse dynamics when `mujoco` is installed,
otherwise pure landmark kinetics (Butterworth + Savitzky–Golay).
MediaPipe world: Y up; timestamps preserved via resampling.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Zero-phase Butterworth low-pass along time (axis 0)."""
    if data.shape[0] < 4:
        return np.array(data, copy=True)
    nyq = 0.5 * fs
    if cutoff >= nyq * 0.99:
        cutoff = nyq * 0.45
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)


def resample_landmarks_uniform(data: np.ndarray, t_src: np.ndarray):
    """Interpolate [frames, 33, 3] onto uniform time from t_src[0]..t_src[-1]."""
    num_frames = data.shape[0]
    t0, t1 = float(t_src[0]), float(t_src[-1])
    duration = t1 - t0
    if duration < 1e-9:
        dt = 1.0
        return np.array(data, copy=True), np.array(t_src, copy=True), dt

    t_uniform = np.linspace(t0, t1, num_frames)
    data_u = np.zeros_like(data, dtype=float)
    for i in range(33):
        for k in range(3):
            data_u[:, i, k] = np.interp(t_uniform, t_src, data[:, i, k])
    dt = duration / max(num_frames - 1, 1)
    return data_u, t_uniform, dt


def get_segment_metrics(p_proximal, p_distal, dt, window_size):
    vector = p_distal - p_proximal
    norm = np.linalg.norm(vector, axis=1)[:, None]
    norm = np.where(norm < 1e-9, 1e-9, norm)
    unit_vector = vector / norm

    pitch = np.degrees(
        np.arctan2(
            np.sqrt(unit_vector[:, 0] ** 2 + unit_vector[:, 2] ** 2),
            np.clip(unit_vector[:, 1], -1.0, 1.0),
        )
    )
    yaw = np.degrees(np.arctan2(unit_vector[:, 0], unit_vector[:, 2]))
    orientations = np.stack([pitch, yaw], axis=1)
    accel = savgol_filter(p_distal, window_size, polyorder=3, deriv=2, delta=dt, axis=0)

    n = unit_vector.shape[0]
    du = np.zeros_like(unit_vector)
    if n >= 2:
        du[0] = (unit_vector[1] - unit_vector[0]) / dt
        du[-1] = (unit_vector[-1] - unit_vector[-2]) / dt
    if n > 2:
        du[1:-1] = (unit_vector[2:] - unit_vector[:-2]) / (2 * dt)
    ang_vel = np.cross(unit_vector, du)

    return {
        "orientation": orientations.tolist(),
        "acceleration": accel.tolist(),
        "angular_velocity": ang_vel.tolist(),
    }


def get_point_linear_kinematics(position: np.ndarray, dt: float, window_size: int):
    vel = savgol_filter(position, window_size, polyorder=3, deriv=1, delta=dt, axis=0)
    accel = savgol_filter(position, window_size, polyorder=3, deriv=2, delta=dt, axis=0)
    return vel, accel


def _preprocess(
    landmarks_sequence,
    fps: int,
    timestamps: np.ndarray | None,
):
    data = np.asarray(landmarks_sequence, dtype=float)
    num_frames = data.shape[0]
    if timestamps is None or len(timestamps) != num_frames:
        t_src = np.arange(num_frames, dtype=float) / float(max(fps, 1))
    else:
        t_src = np.asarray(timestamps, dtype=float)

    data_u, t_uniform, dt = resample_landmarks_uniform(data, t_src)
    duration = float(t_uniform[-1] - t_uniform[0]) if num_frames > 1 else 0.0
    fs_eff = (num_frames - 1) / duration if duration > 1e-9 else float(max(fps, 1))

    processed = np.zeros_like(data_u)
    for i in range(33):
        processed[:, i, :] = butter_lowpass_filter(data_u[:, i, :], cutoff=6, fs=fs_eff)

    window = min(15, num_frames // 2 * 2 - 1)
    if window < 5:
        window = 5

    # Resampled positions without per-landmark low-pass — used for vGRF accelerations
    # (6 Hz Butterworth strongly damps vertical bounce needed for realistic d²y/dt²).
    return processed, data_u, t_src, dt, fs_eff, window


def _landmark_segment_payload(processed_data: np.ndarray, dt: float, window: int):
    hip_mid = (processed_data[:, 23, :] + processed_data[:, 24, :]) / 2
    hip_vel, hip_accel = get_point_linear_kinematics(hip_mid, dt, window)

    pelvis_seg = get_segment_metrics(
        processed_data[:, 24, :], processed_data[:, 23, :], dt, window
    )
    pelvis_seg["acceleration"] = hip_accel.tolist()

    r_thigh_prox, r_thigh_dist = processed_data[:, 24, :], processed_data[:, 26, :]
    r_shank_prox, r_shank_dist = processed_data[:, 26, :], processed_data[:, 28, :]
    r_foot_prox, r_foot_dist = processed_data[:, 28, :], processed_data[:, 32, :]

    thigh_metrics = get_segment_metrics(r_thigh_prox, r_thigh_dist, dt, window)
    shank_metrics = get_segment_metrics(r_shank_prox, r_shank_dist, dt, window)
    foot_metrics = get_segment_metrics(r_foot_prox, r_foot_dist, dt, window)
    return (
        hip_mid,
        hip_vel,
        hip_accel,
        pelvis_seg,
        thigh_metrics,
        shank_metrics,
        foot_metrics,
    )


def _response_metadata(fs_eff: float, fps: int, window: int, engine: str):
    return {
        "fps": fps,
        "effective_fs_hz": round(float(fs_eff), 3),
        "filter_butterworth": "6Hz Low-pass",
        "filter_savgol": f"Window {window}, Poly 3",
        "engine": engine,
    }


def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps, timestamps=None):
    height_cm = height_cm or 0.0
    data = np.asarray(landmarks_sequence, dtype=float)
    num_frames = data.shape[0]
    if num_frames < 5:
        raise ValueError("Not enough frames for analysis. Minimum 5 frames required.")

    t_in = None if timestamps is None else np.asarray(timestamps, dtype=float)
    processed, resampled_raw, t_src, dt, fs_eff, window = _preprocess(
        landmarks_sequence, fps, t_in
    )

    (
        hip_mid,
        hip_vel,
        hip_accel,
        pelvis_seg,
        thigh_metrics,
        shank_metrics,
        foot_metrics,
    ) = _landmark_segment_payload(processed, dt, window)

    shank_av = np.array(shank_metrics["angular_velocity"])
    shank_av_mag = np.linalg.norm(shank_av, axis=1)

    mj_frames = None
    try:
        from mujoco_pipeline import run_mujoco_inverse_dynamics

        mj_frames = run_mujoco_inverse_dynamics(
            processed,
            dt,
            float(weight_kg),
            float(height_cm),
            int(fps),
            t_src,
            landmarks_for_vgrf=resampled_raw,
        )
    except Exception:
        mj_frames = None

    frames = []
    for i in range(num_frames):
        ts = float(t_src[i]) if i < len(t_src) else float(i) / float(max(fps, 1))

        seg_block = {
            "pelvis": {
                "acceleration": pelvis_seg["acceleration"][i],
                "angular_velocity": pelvis_seg["angular_velocity"][i],
                "orientation": pelvis_seg["orientation"][i],
            },
            "thigh": {
                "acceleration": thigh_metrics["acceleration"][i],
                "angular_velocity": thigh_metrics["angular_velocity"][i],
                "orientation": thigh_metrics["orientation"][i],
            },
            "shank": {
                "acceleration": shank_metrics["acceleration"][i],
                "angular_velocity": shank_metrics["angular_velocity"][i],
                "orientation": shank_metrics["orientation"][i],
            },
            "foot": {
                "acceleration": foot_metrics["acceleration"][i],
                "angular_velocity": foot_metrics["angular_velocity"][i],
                "orientation": foot_metrics["orientation"][i],
            },
        }

        if mj_frames and i < len(mj_frames):
            f = dict(mj_frames[i])
            f.update(seg_block)
            frames.append(f)
        else:
            pelvis_accel_y = float(hip_accel[i][1]) if i < len(hip_accel) else 0.0
            fy_raw = float(weight_kg) * (pelvis_accel_y + 9.81)
            fy_dsp = max(0.0, fy_raw)
            half = fy_dsp / 2.0
            frames.append(
                {
                    "timestamp": ts,
                    "frame_idx": i,
                    "vertical_force": fy_raw,
                    "com_position": hip_mid[i].tolist(),
                    "com_velocity": hip_vel[i].tolist(),
                    "grf_left": [0.0, half, 0.0],
                    "grf_right": [0.0, half, 0.0],
                    "residual_error": 0.0,
                    "warnings": [],
                    **seg_block,
                }
            )

    engine = "mujoco-inverse-dynamics" if mj_frames else "landmark-kinetics-v2"
    if mj_frames:
        max_force = float(max(abs(f["vertical_force"]) for f in mj_frames))
    else:
        max_force = float(
            np.max(
                np.maximum(
                    float(weight_kg) * (hip_accel[:, 1] + 9.81),
                    0.0,
                )
            )
        )

    return {
        "metadata": _response_metadata(fs_eff, fps, window, engine),
        "frames": frames,
        "summary": {
            "total_frames": num_frames,
            "solve_time_s": 0.0,
            "mean_residual_m": 0.0,
            "max_residual_m": 0.0,
            "total_warnings": 0,
            "fps": fps,
            "max_force_newtons": round(max_force, 2),
            "max_shank_omega_rad_s": round(float(np.max(shank_av_mag)), 3),
        },
    }
