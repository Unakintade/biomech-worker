"""SMPL 24-joint indices (0-based), Y-up world coordinates (meters)."""

from __future__ import annotations

# Standard SMPL kinematic order
PELVIS = 0
L_HIP, R_HIP = 1, 2
SPINE1 = 3
L_KNEE, R_KNEE = 4, 5
SPINE2 = 6
L_ANKLE, R_ANKLE = 7, 8
SPINE3 = 9
L_FOOT, R_FOOT = 10, 11
# 12 Neck, 13 L_Collar, 14 R_Collar, 15 Head, 16–17 L/R_Shoulder,
# 18–19 L/R_Elbow, 20–21 L/R_Wrist, 22–23 L/R_Hand

# Use foot joint as toe proxy for stance (SMPL has no toe landmarks)
L_TOE, R_TOE = L_FOOT, R_FOOT

N_SMPL_JOINTS = 24


def assert_joints_shape(joints: np.ndarray) -> None:
    if (
        joints.ndim != 3
        or joints.shape[1] != N_SMPL_JOINTS
        or joints.shape[2] != 3
    ):
        raise ValueError(
            f"joints_world must be (T, {N_SMPL_JOINTS}, 3); got {getattr(joints, 'shape', None)}"
        )
