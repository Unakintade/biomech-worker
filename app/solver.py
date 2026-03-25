import numpy as np
import os
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.spatial.transform import Rotation as R

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Applies a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Use filtfilt for zero-phase shift (preserves timing of peaks)
    y = filtfilt(b, a, data, axis=0)
    return y

def get_segment_metrics(p_proximal, p_distal, dt, window_size):
    """
    Calculates Orientation, Angular Velocity, and Linear Acceleration 
    for a body segment defined by two points.
    """
    # 1. Orientation (Euler Angles)
    # Vector representing the segment limb
    vector = p_distal - p_proximal
    unit_vector = vector / np.linalg.norm(vector, axis=1)[:, None]
    
    # Calculate pitch and yaw (simplified relative to vertical)
    pitch = np.degrees(np.arctan2(unit_vector[:, 0], unit_vector[:, 2]))
    yaw = np.degrees(np.arctan2(unit_vector[:, 1], unit_vector[:, 2]))
    orientations = np.stack([pitch, yaw], axis=1)

    # 2. Linear Acceleration (2nd derivative of distal point)
    # distal point represents the 'end' of the segment (e.g. the ankle for the shank)
    accel = savgol_filter(p_distal, window_size, polyorder=3, deriv=2, delta=dt, axis=0)
    
    # 3. Angular Velocity (Derivative of orientation)
    # We use SavGol on the vector components to find rotation rate
    ang_vel = savgol_filter(orientations, window_size, polyorder=3, deriv=1, delta=dt, axis=0)
    
    return {
        "orientation": orientations.tolist(),
        "acceleration": accel.tolist(),
        "angular_velocity": ang_vel.tolist()
    }

def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps):
    """
    High-fidelity Kinematics Pipeline (Hyfydy-style outputs)
    Uses Butterworth + Savitzky-Golay filtering to estimate IMU-grade data.
    """
    dt = 1.0 / fps
    
    # landmarks_sequence: [frames, 33_landmarks, 3_coords]
    data = np.array(landmarks_sequence)
    num_frames = data.shape[0]

    # --- STAGE 1: BUTTERWORTH FILTERING (Noise Reduction) ---
    # Standard biomechanical cutoff: 6Hz
    processed_data = np.zeros_like(data)
    for i in range(33): # For each landmark
        processed_data[:, i, :] = butter_lowpass_filter(data[:, i, :], cutoff=6, fs=fps)

    # --- STAGE 2: SEGMENT DEFINITIONS ---
    # We use MediaPipe indices to define our segments
    # Pelvis: Midpoint of Hips (23, 24)
    pelvis = (processed_data[:, 23, :] + processed_data[:, 24, :]) / 2.0
    
    # Thigh: Hip (24) to Knee (26)
    r_thigh_prox, r_thigh_dist = processed_data[:, 24, :], processed_data[:, 26, :]
    
    # Shank: Knee (26) to Ankle (28)
    r_shank_prox, r_shank_dist = processed_data[:, 26, :], processed_data[:, 28, :]
    
    # Foot: Ankle (28) to Toe (32)
    r_foot_prox, r_foot_dist = processed_data[:, 28, :], processed_data[:, 32, :]

    # --- STAGE 3: SAVITZKY-GOLAY & METRIC EXTRACTION ---
    # Window size must be odd
    window = min(15, num_frames // 2 * 2 - 1)
    if window < 5: window = 5

    # Extract metrics for requested segments
    # (Using Right side as the primary example)
    pelvis_metrics = get_segment_metrics(processed_data[:, 11], processed_data[:, 23], dt, window) # Spine to Pelvis
    thigh_metrics = get_segment_metrics(r_thigh_prox, r_thigh_dist, dt, window)
    shank_metrics = get_segment_metrics(r_shank_prox, r_shank_dist, dt, window)
    foot_metrics = get_segment_metrics(r_foot_prox, r_foot_dist, dt, window)

    # Calculate Global Force using Pelvis Acceleration
    # F = m(a + g)
    pelvis_accel_z = np.array(pelvis_metrics["acceleration"])[:, 2]
    forces = weight_kg * (np.abs(pelvis_accel_z) + 9.81)

    timestamps = [float(i * dt) for i in range(num_frames)]

    return {
        "metadata": {
            "fps": fps,
            "filter_butterworth": "6Hz Low-pass",
            "filter_savgol": f"Window {window}, Poly 3"
        },
        "timestamps": timestamps,
        "segments": {
            "pelvis": pelvis_metrics,
            "thigh": thigh_metrics,
            "shank": shank_metrics,
            "foot": foot_metrics
        },
        "vertical_forces": forces.tolist(),
        "summary": {
            "max_force_newtons": round(float(np.max(forces)), 2),
            "max_shank_ang_vel": round(float(np.max(np.abs(shank_metrics["angular_velocity"]))), 2)
        }
    }
