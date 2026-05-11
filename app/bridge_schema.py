"""
Expected JSON from the Colab GPU service (POST /process-sprint).

Colab should run mmhuman3d (SMPL), return 3D joints in a Y-up, meter-scale
world frame aligned with app/models/biped_sprint.xml (gravity along -Y).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .smpl_joints import assert_joints_shape

JointFrame = Literal["world_y_up", "camera_flip_y"]


def apply_joint_coordinate_frame(joints: np.ndarray, frame: str) -> np.ndarray:
    """
    world_y_up: no change (matches biped_sprint.xml gravity -Y).

    camera_flip_y: negate world Y (common when camera / OpenGL style differs from MJCF).
    """
    j = np.asarray(joints, dtype=np.float64, copy=True)
    f = (frame or "world_y_up").lower().strip()
    if f in ("camera_flip_y", "flip_y", "opengl_y_up"):
        j[..., 1] *= -1.0
    return j


class ColabSprintResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    smpl_betas: list[float] = Field(default_factory=list)
    """SMPL shape parameters (typically 10); may be omitted or partial."""

    joints_world: list[list[list[float]]]
    """Sequence of SMPL-24 joint positions: (T, 24, 3) in meters, Y-up."""

    fps: float | None = None
    timestamps: list[float] | None = None

    joints_coordinate_frame: JointFrame = "world_y_up"
    """
    world_y_up: joints already match MJCF (Y up, meters).

    camera_flip_y: negate Y on every joint (use if mmhuman3d / camera coords look inverted).
    """

    @field_validator("joints_world")
    @classmethod
    def _non_empty_sequence(cls, v: list[list[list[float]]]) -> list[list[list[float]]]:
        if not v:
            raise ValueError("joints_world must be non-empty")
        return v

    def joints_numpy(self) -> np.ndarray:
        arr = np.asarray(self.joints_world, dtype=np.float64)
        assert_joints_shape(arr)
        return arr

    def joints_numpy_mujoco(self) -> np.ndarray:
        """Joints after optional coordinate fix for biped_sprint.xml."""
        return apply_joint_coordinate_frame(self.joints_numpy(), str(self.joints_coordinate_frame))

    def betas_numpy(self) -> np.ndarray:
        b = np.asarray(self.smpl_betas, dtype=np.float64).ravel()
        if b.size == 0:
            return np.zeros(10, dtype=np.float64)
        if b.size < 10:
            out = np.zeros(10, dtype=np.float64)
            out[: b.size] = b
            return out
        return b[:10]

    def time_array(self, fallback_dt: float) -> np.ndarray:
        if self.timestamps is not None and len(self.timestamps) == len(self.joints_world):
            return np.asarray(self.timestamps, dtype=np.float64)
        n = len(self.joints_world)
        return np.arange(n, dtype=np.float64) * float(fallback_dt)


def parse_colab_result(raw: dict[str, Any]) -> ColabSprintResult:
    return ColabSprintResult.model_validate(raw)
