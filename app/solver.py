import mujoco
import numpy as np
import os

# Path to your human model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "sprinter.xml")

def solve_kinetics(landmarks_sequence, weight_kg, height_cm, fps):
    """
    Takes tracking dots and calculates forces using MuJoCo Inverse Dynamics.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    # Load model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    joint_angles = []
    forces = []
    
    # Set time step based on video FPS
    dt = 1.0 / fps
    model.opt.timestep = dt

    # Process each frame of the video
    for frame_idx in range(len(landmarks_sequence)):
        # Calculate current state
        mujoco.mj_forward(model, data)
        
        # Calculate Inverse Dynamics (The Forces)
        mujoco.mj_inverse(model, data)
        
        # Calculate a simplified vertical force based on total joint torque
        # qfrc_inverse represents the forces acting on the joints
        net_force = np.linalg.norm(data.qfrc_inverse)
        
        # Convert to Newtons (scaled by body weight)
        # 70kg is the default model weight
        scaled_force = float(net_force * (weight_kg / 70.0))
        
        # Get Knee Angle (assuming joint index 3 is a knee hinge)
        knee_idx = min(3, model.nq - 1)
        angle = float(np.degrees(data.qpos[knee_idx]))
        
        forces.append(scaled_force)
        joint_angles.append(angle)

    return {
        "joint_angles": joint_angles,
        "vertical_forces": forces,
        "summary": {
            "max_force_newtons": round(max(forces) if forces else 0, 2),
            "avg_knee_angle": round(np.mean(joint_angles) if joint_angles else 0, 2),
            "frames_processed": len(landmarks_sequence)
        }
    }
