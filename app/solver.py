import mujoco
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sprinter.xml")

def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps):
    """
    Takes tracking dots and calculates forces by MOVING the model
    frame-by-frame and calculating the required physics.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    joint_angles = []
    forces = []
    
    # Set time step based on video FPS
    dt = 1.0 / fps
    model.opt.timestep = dt

    # 1. We need to know which MediaPipe dots map to which joints.
    # For this basic model, we'll map the Hip and Knee angles.
    # landmarks_sequence shape is [frames, 33_landmarks, 3_coords]
    
    for frame_idx in range(len(landmarks_sequence)):
        frame_dots = landmarks_sequence[frame_idx]
        
        # --- KINEMATIC MAPPING ---
        # MediaPipe Landmark Indices:
        # Left Hip: 23, Left Knee: 25, Left Ankle: 27
        # Right Hip: 24, Right Knee: 26, Right Ankle: 28
        
        # Calculate Right Knee Angle using vectors (Trigonometry)
        # Vector Hip -> Knee
        v1 = frame_dots[26] - frame_dots[24] 
        # Vector Knee -> Ankle
        v2 = frame_dots[28] - frame_dots[26]
        
        # Calculate angle between vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 > 0 and norm_v2 > 0:
            angle_rad = np.arccos(np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = 0

        # --- UPDATE MUJOCO STATE ---
        # We tell MuJoCo: "The knee is now at this angle"
        # In our sprinter.xml, joint 3 is the right knee hinge
        if model.nq > 3:
            data.qpos[3] = np.radians(angle_deg) 

        # --- CALCULATE PHYSICS ---
        # Update the positions of all limbs based on the new joint angle
        mujoco.mj_forward(model, data)
        
        # Run Inverse Dynamics: "How much force was needed to move the knee like that?"
        mujoco.mj_inverse(model, data)
        
        # Extract the force (torque) acting on the knee joint
        knee_torque = abs(data.qfrc_inverse[3])
        
        # Scale by weight to get a "Force-Plate-like" estimate in Newtons
        # Simplified: Torque / lever_arm * weight_ratio
        estimated_force = knee_torque * (weight_kg / 70.0)
        
        joint_angles.append(float(angle_deg))
        forces.append(float(estimated_force))

    return {
        "joint_angles": joint_angles,
        "vertical_forces": forces,
        "summary": {
            "max_force_newtons": round(max(forces) if forces else 0, 2),
            "avg_knee_angle": round(np.mean(joint_angles) if joint_angles else 0, 2),
            "frames_processed": len(landmarks_sequence)
        }
    }
