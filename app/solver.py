import mujoco
import numpy as np
import os
from scipy.signal import savgol_filter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sprinter.xml")

def get_angle(p1, p2, p3):
    """Calculates the angle at p2 given three points p1, p2, p3."""
    v1 = p1 - p2
    v2 = p3 - p2
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 > 0 and norm_v2 > 0:
        return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)))
    return 0

def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps):
    """
    Calculates bilateral kinematics (both legs) and vertical forces.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Result containers
    results_history = {
        "right_knee": [],
        "left_knee": [],
        "right_hip": [],
        "left_hip": [],
        "forces": []
    }
    
    dt = 1.0 / fps
    model.opt.timestep = dt
    qpos_history = []

    # 1. Map Landmarks to Joint Positions
    for frame_idx in range(len(landmarks_sequence)):
        fd = landmarks_sequence[frame_idx]
        current_qpos = np.zeros(model.nq)
        
        # --- ROOT POSITION (Global movement) ---
        hip_center = (fd[23] + fd[24]) / 2.0
        scale_factor = (height_cm / 100.0)
        current_qpos[0:3] = [
            (hip_center[0] - 0.5) * scale_factor, 
            -(hip_center[1] - 0.5) * scale_factor, 
            (1.2 - hip_center[1]) * scale_factor
        ]
        current_qpos[3:7] = [1, 0, 0, 0]

        # --- CALCULATE ANGLES ---
        # Right Leg: Hip(24), Knee(26), Ankle(28)
        r_knee_angle = get_angle(fd[24], fd[26], fd[28])
        # Left Leg: Hip(23), Knee(25), Ankle(27)
        l_knee_angle = get_angle(fd[23], fd[25], fd[27])
        
        # Hip angles (relative to torso - simplified)
        r_hip_angle = get_angle(fd[12], fd[24], fd[26])
        l_hip_angle = get_angle(fd[11], fd[23], fd[25])

        # --- UPDATE MUJOCO QPOS ---
        # Mapping based on sprinter.xml structure:
        # r_hip (ball): 7,8,9 | r_knee: 10
        # l_hip (ball): 11,12,13 | l_knee: 14
        if model.nq > 14:
            current_qpos[10] = np.radians(r_knee_angle)
            current_qpos[14] = np.radians(l_knee_angle)
            # Hip ball joints are complex, we'll map a single axis for this demo
            current_qpos[7] = np.radians(r_hip_angle - 180) 
            current_qpos[11] = np.radians(l_hip_angle - 180)

        qpos_history.append(current_qpos)

    # 2. SMOOTHING
    qpos_history = np.array(qpos_history)
    window = min(11, len(qpos_history) // 2 * 2 - 1)
    if window > 3:
        for i in range(model.nq):
            qpos_history[:, i] = savgol_filter(qpos_history[:, i], window, 3)

    # 3. KINETICS PASS
    mass = mujoco.mj_getTotalmass(model)
    weight_scale = weight_kg / mass

    for t in range(len(qpos_history)):
        if t < 2:
            results_history["right_knee"].append(float(np.degrees(qpos_history[t][10])))
            results_history["left_knee"].append(float(np.degrees(qpos_history[t][14])))
            results_history["forces"].append(weight_kg * 9.81)
            continue

        data.qpos[:] = qpos_history[t]
        data.qvel[:] = (qpos_history[t][:model.nv] - qpos_history[t-1][:model.nv]) / dt
        prev_qvel = (qpos_history[t-1][:model.nv] - qpos_history[t-2][:model.nv]) / dt
        data.qacc[:] = (data.qvel - prev_qvel) / dt

        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)
        
        # Vertical force = ma_z + gravity
        total_force = (abs(data.qfrc_inverse[2]) + (mass * 9.81)) * weight_scale
        
        results_history["right_knee"].append(float(np.degrees(data.qpos[10])))
        results_history["left_knee"].append(float(np.degrees(data.qpos[14])))
        results_history["forces"].append(float(total_force))

    return {
        "joint_angles": {
            "right_knee": results_history["right_knee"],
            "left_knee": results_history["left_knee"]
        },
        "vertical_forces": results_history["forces"],
        "summary": {
            "max_force_newtons": round(max(results_history["forces"]), 2),
            "avg_right_knee": round(np.mean(results_history["right_knee"]), 2),
            "avg_left_knee": round(np.mean(results_history["left_knee"]), 2)
        }
    }
