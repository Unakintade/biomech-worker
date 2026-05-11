"""
Expected JSON from the Colab GPU service (POST /process-sprint).

Colab should run mmhuman3d (SMPL), return 3D joints in a Y-up, meter-scale
world frame aligned with app/models/biped_sprint.xml (gravity along -Y).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .smpl_joints import assert_joints_shape


class ColabSprintResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    smpl_betas: list[float] = Field(default_factory=list)
    """SMPL shape parameters (typically 10); may be omitted or partial."""

    joints_world: list[list[list[float]]]
    """Sequence of SMPL-24 joint positions: (T, 24, 3) in meters, Y-up."""

    fps: float | None = None
    timestamps: list[float] | None = None

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
