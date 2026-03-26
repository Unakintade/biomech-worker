"""
Landmark-based kinetics (butterworth + Savitzky–Golay on resampled time).
Uses real per-frame timestamps when irregular; MediaPipe world frame (Y up).
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


def resample_landmarks_uniform(data: np.ndarray, t_src: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate [frames, 33, 3] onto uniform time from t_src[0]..t_src[-1]. Returns data_u, t_uniform, dt."""
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
    """
    Segment direction u = normalize(distal - proximal).
    - Linear acceleration of distal point (SG 2nd deriv).
    - Angular velocity ω ≈ u × (du/dt) (rad/s) for |u| ≈ 1.
    - Orientation pitch/yaw helpers from u (Y-up world frame).
    """
    vector = p_distal - p_proximal
    norm = np.linalg.norm(vector, axis=1)[:, None]
    norm = np.where(norm < 1e-9, 1e-9, norm)
    unit_vector = vector / norm

    # Angle from +Y (vertical) and yaw in X–Z
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


def get_point_linear_kinematics(position: np.ndarray, dt: float, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Linear velocity and acceleration of a 3D point track (n, 3)."""
    vel = savgol_filter(position, window_size, polyorder=3, deriv=1, delta=dt, axis=0)
    accel = savgol_filter(position, window_size, polyorder=3, deriv=2, delta=dt, axis=0)
    return vel, accel


def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps, timestamps=None):
    """
    High-fidelity kinematics on MediaPipe world landmarks.
    - Resamples to uniform time using actual timestamps when provided.
    - Pelvis linear kinetics use hip midpoint (23 + 24) / 2; vertical axis is Y (up).
    - Pelvis orientation uses hip line right→left (24 → 23).
    height_cm reserved for future scaling (unused).
    """
    _ = height_cm  # API compatibility
    data = np.asarray(landmarks_sequence, dtype=float)
    num_frames = data.shape[0]

    if num_frames < 5:
        raise ValueError("Not enough frames for analysis. Minimum 5 frames required.")

    if timestamps is None or len(timestamps) != num_frames:
        t_src = np.arange(num_frames, dtype=float) / float(max(fps, 1))
    else:
        t_src = np.asarray(timestamps, dtype=float)

    data_u, t_uniform, dt = resample_landmarks_uniform(data, t_src)
    duration = float(t_uniform[-1] - t_uniform[0]) if num_frames > 1 else 0.0
    fs_eff = (num_frames - 1) / duration if duration > 1e-9 else float(max(fps, 1))

    processed_data = np.zeros_like(data_u)
    for i in range(33):
        processed_data[:, i, :] = butter_lowpass_filter(data_u[:, i, :], cutoff=6, fs=fs_eff)

    window = min(15, num_frames // 2 * 2 - 1)
    if window < 5:
        window = 5

    hip_mid = (processed_data[:, 23, :] + processed_data[:, 24, :]) / 2
    hip_vel, hip_accel = get_point_linear_kinematics(hip_mid, dt, window)

    # Pelvis: right hip (24) to left hip (23) for segment angular metrics;
    # expose hip-mid linear acceleration under pelvis for consistency with CoM proxy.
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

    # Vertical support estimate: F_y ≈ m (a_y + g), Y up
    pelvis_accel_y = hip_accel[:, 1]
    forces = weight_kg * (pelvis_accel_y + 9.81)
    forces_display = np.maximum(forces, 0.0)

    frames = []
    for i in range(num_frames):
        ts = float(t_src[i]) if i < len(t_src) else float(i) / float(max(fps, 1))
        fy = float(forces_display[i])
        half = fy / 2.0
        frames.append(
            {
                "timestamp": ts,
                "frame_idx": i,
                "vertical_force": float(forces[i]),
                "com_position": hip_mid[i].tolist(),
                "com_velocity": hip_vel[i].tolist(),
                "grf_left": [0.0, half, 0.0],
                "grf_right": [0.0, half, 0.0],
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
        )

    shank_av = np.array(shank_metrics["angular_velocity"])
    shank_av_mag = np.linalg.norm(shank_av, axis=1)

    return {
        "metadata": {
            "fps": fps,
            "effective_fs_hz": round(float(fs_eff), 3),
            "filter_butterworth": "6Hz Low-pass",
            "filter_savgol": f"Window {window}, Poly 3",
            "engine": "landmark-kinetics-v2",
        },
        "frames": frames,
        "summary": {
            "total_frames": num_frames,
            "solve_time_s": 0.0,
            "mean_residual_m": 0.0,
            "max_residual_m": 0.0,
            "total_warnings": 0,
            "fps": fps,
            "max_force_newtons": round(float(np.max(forces_display)), 2),
            "max_shank_omega_rad_s": round(float(np.max(shank_av_mag)), 3),
        },
    }
