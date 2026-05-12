"""SMPL 24-joint indices (0-based), Y-up world coordinates (meters)."""

from __future__ import annotations

import numpy as np

# Standard SMPL kinematic order
PELVIS = 0
L_HIP, R_HIP = 1, 2
SPINE1 = 3
L_KNEE, R_KNEE = 4, 5
SPINE2 = 6
L_ANKLE, R_ANKLE = 7, 8
SPINE3 = 9
L_FOOT, R_FOOT = 10, 11
NECK = 12
L_COLLAR, R_COLLAR = 13, 14
HEAD = 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23

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
