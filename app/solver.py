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
    Calculates bilateral kinematics and estimates Ground Reaction Forces
    using Center of Mass (CoM) acceleration.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    results_history = {
        "right_knee": [],
        "left_knee": [],
        "forces": []
    }
    
    dt = 1.0 / fps
    model.opt.timestep = dt
    qpos_history = []

    # 1. Map Landmarks to Joint Positions
    # We use a standard scale factor based on the subject's height
    scale_factor = (height_cm / 100.0)

    for frame_idx in range(len(landmarks_sequence)):
        fd = landmarks_sequence[frame_idx]
        current_qpos = np.zeros(model.nq)
        
        # --- ROOT POSITION (Global movement) ---
        # We track the hips to move the entire model in 3D space
        hip_center = (fd[23] + fd[24]) / 2.0
        
        # X: Left/Right, Y: Forward/Back, Z: Vertical (Up)
        # MediaPipe Y is inverted (0 is top), so we flip it for MuJoCo
        current_qpos[0:3] = [
            (hip_center[0] - 0.5) * scale_factor, 
            -(hip_center[2]), # Use MediaPipe Z for depth if available
            (1.1 - hip_center[1]) * scale_factor 
        ]
        current_qpos[3:7] = [1, 0, 0, 0] # Identity quaternion

        # --- CALCULATE JOINT ANGLES ---
        r_knee_angle = get_angle(fd[24], fd[26], fd[28])
        l_knee_angle = get_angle(fd[23], fd[25], fd[27])
        
        # Map to MuJoCo qpos indices
        # Note: Indexing depends on your specific XML structure
        if model.nq > 14:
            current_qpos[10] = np.radians(r_knee_angle)
            current_qpos[14] = np.radians(l_knee_angle)

        qpos_history.append(current_qpos)

    # 2. TEMPORAL SMOOTHING
    # Reduced window size to preserve acceleration peaks (Impacts)
    qpos_history = np.array(qpos_history)
    window = min(7, len(qpos_history) // 2 * 2 - 1) 
    if window > 3:
        for i in range(model.nq):
            qpos_history[:, i] = savgol_filter(qpos_history[:, i], window, 2)

    # 3. KINETICS PASS (F = m(a+g))
    mass = mujoco.mj_getTotalmass(model)
    weight_scale = weight_kg / mass
    com_vel_history = []

    # First, calculate CoM velocities across all frames
    for t in range(len(qpos_history)):
        data.qpos[:] = qpos_history[t]
        mujoco.mj_forward(model, data)
        # Store 3D Center of Mass position
        com_vel_history.append(np.array(data.subtree_com[0]))

    # Calculate Force using Second Derivative of Center of Mass
    for t in range(len(com_vel_history)):
        if t < 2:
            results_history["right_knee"].append(float(np.degrees(qpos_history[t][10])))
            results_history["left_knee"].append(float(np.degrees(qpos_history[t][14])))
            results_history["forces"].append(weight_kg * 9.81)
            continue

        # Calculate Vertical CoM Acceleration (a_z) via Finite Difference
        v_current = (com_vel_history[t][2] - com_vel_history[t-1][2]) / dt
        v_prev = (com_vel_history[t-1][2] - com_vel_history[t-2][2]) / dt
        accel_z = (v_current - v_prev) / dt

        # Ground Reaction Force (GRF) = mass * (acceleration + gravity)
        # We cap the force at 0 (you can't have negative GRF)
        grf_z = mass * (accel_z + 9.81)
        final_force = max(0, grf_z * weight_scale)
        
        results_history["right_knee"].append(float(np.degrees(qpos_history[t][10])))
        results_history["left_knee"].append(float(np.degrees(qpos_history[t][14])))
        results_history["forces"].append(float(final_force))

    return {
        "joint_angles": {
            "right_knee": results_history["right_knee"],
            "left_knee": results_history["left_knee"]
        },
        "vertical_forces": results_history["forces"],
        "summary": {
            "max_force_newtons": round(max(results_history["forces"]), 2),
            "avg_right_knee": round(np.mean(results_history["right_knee"]), 2),
            "avg_left_knee": round(np.mean(results_history["left_knee"]), 2),
            "frames_processed": len(landmarks_sequence)
        }
    }
