"""Lower-body landmark index sets for SMPL-24 vs MediaPipe 33 (BlazePose)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LandmarkLayout(str, Enum):
    SMPL24 = "smpl24"
    MEDIAPIPE33 = "mediapipe33"


@dataclass(frozen=True)
class LowerBodyIndices:
    l_hip: int
    r_hip: int
    l_knee: int
    r_knee: int
    l_ankle: int
    r_ankle: int
    l_toe: int
    r_toe: int


def lower_body_indices(layout: LandmarkLayout) -> LowerBodyIndices:
    if layout == LandmarkLayout.SMPL24:
        from .smpl_joints import L_ANKLE, L_FOOT, L_HIP, L_KNEE, R_ANKLE, R_FOOT, R_HIP, R_KNEE

        return LowerBodyIndices(L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)
    # BlazePose world landmarks (same as legacy stride-kin-solver / mujoco_pipeline)
    return LowerBodyIndices(23, 24, 25, 26, 27, 28, 31, 32)


def min_joints_for_layout(layout: LandmarkLayout) -> int:
    return 33 if layout == LandmarkLayout.MEDIAPIPE33 else 24
