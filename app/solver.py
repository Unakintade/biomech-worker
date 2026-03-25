import mujoco
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sprinter.xml")

def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps):
    """
    Takes tracking dots and calculates forces by MOVING the model
    frame-by-frame and calculating the required physics derivatives.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    joint_angles = []
    forces = []
    
    dt = 1.0 / fps
    model.opt.timestep = dt

    # To calculate forces, we need velocity and acceleration (derivatives of position)
    # We store a history of positions to calculate these.
    qpos_history = []

    # 1. First Pass: Convert all landmarks to MuJoCo joint positions (qpos)
    for frame_idx in range(len(landmarks_sequence)):
        frame_dots = landmarks_sequence[frame_idx]
        
        # Create a qpos array for this frame [root_pos(3), root_quat(4), joints...]
        current_qpos = np.zeros(model.nq)
        
        # --- ROOT POSITION (Hips) ---
        # Map MediaPipe Hip Center (avg of 23 and 24) to MuJoCo Root
        hip_center = (frame_dots[23] + frame_dots[24]) / 2.0
        # Scale normalized coordinates to meters (rough estimate based on height)
        scale_factor = (height_cm / 100.0) 
        current_qpos[0:3] = [
            (hip_center[0] - 0.5) * scale_factor, 
            -(hip_center[1] - 0.5) * scale_factor, 
            (1.0 - hip_center[1]) * scale_factor # Vertical height
        ]
        # Set default quaternion (no rotation for now)
        current_qpos[3:7] = [1, 0, 0, 0]

        # --- KNEE ANGLE CALCULATION ---
        # Vector Hip -> Knee (Right side: 24 -> 26)
        v1 = frame_dots[26] - frame_dots[24] 
        # Vector Knee -> Ankle (Right side: 26 -> 28)
        v2 = frame_dots[28] - frame_dots[26]
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            angle_rad = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
            # Joint index for right knee in our XML is typically index 10 
            # (Root=7 + Hip_Ball=3 = 10)
            if model.nq > 10:
                current_qpos[10] = angle_rad
        
        qpos_history.append(current_qpos)

    # 2. Second Pass: Calculate derivatives and run mj_inverse
    for t in range(len(qpos_history)):
        if t < 2: # Skip first two frames to allow for finite difference
            joint_angles.append(float(np.degrees(qpos_history[t][10]) if model.nq > 10 else 0))
            forces.append(0.0)
            continue

        # Set Current Position
        data.qpos[:] = qpos_history[t]

        # Numerical Differentiation for Velocity (v = dp/dt)
        data.qvel[:] = (qpos_history[t][:model.nv] - qpos_history[t-1][:model.nv]) / dt
        
        # Numerical Differentiation for Acceleration (a = dv/dt)
        prev_qvel = (qpos_history[t-1][:model.nv] - qpos_history[t-2][:model.nv]) / dt
        data.qacc[:] = (data.qvel - prev_qvel) / dt

        # --- CALCULATE PHYSICS ---
        # Updates geometry based on qpos
        mujoco.mj_forward(model, data)
        
        # Runs Inverse Dynamics: calculates forces needed to achieve the current qacc
        mujoco.mj_inverse(model, data)
        
        # qfrc_inverse[2] is the Z-axis (vertical) force at the root
        # This represents the net vertical force required to move the body mass
        vertical_force_net = data.qfrc_inverse[2]
        
        # Add gravity compensation (F = ma + mg)
        mass = mujoco.mj_getTotalmass(model)
        total_vertical_force = abs(vertical_force_net) + (mass * 9.81)
        
        # Scale to user weight
        weight_scale = weight_kg / mass
        final_force = total_vertical_force * weight_scale
        
        joint_angles.append(float(np.degrees(data.qpos[10]) if model.nq > 10 else 0))
        forces.append(float(final_force))

    return {
        "joint_angles": joint_angles,
        "vertical_forces": forces,
        "summary": {
            "max_force_newtons": round(max(forces) if forces else 0, 2),
            "avg_knee_angle": round(np.mean(joint_angles) if joint_angles else 0, 2),
            "frames_processed": len(landmarks_sequence)
        }
    }
