"""
Two-mass vertical model for sprinting vGRF (sagittal / vertical dynamics).

Treats the runner as:
  - m_u: upper lump (HAT + pelvis + swing-side mass lumped with trunk proxy) — motion from hip height
  - m_s: stance-limb lump — vertical motion from the stance leg COM proxy (hip–knee–ankle mean)

During single support (one foot on ground from MuJoCo contacts):
    F_y ≈ m_u * (a_hip + g) + m_s * (a_stance_leg + g)

During double support, collapse to whole-body proxy:
    F_y ≈ M * (a_hip + g)

Flight (no foot contact): F_y = 0.

Mass fractions follow typical segment tables (~one leg ≈ 16% body mass, remainder ≈ upper lump).
Y-up world frame (MediaPipe / current MJCF).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

G = 9.81

# One leg (thigh+shank+foot) ~16% M (De Leva–style); rest assigned to upper lump for this vertical split
M_STANCE_LEG_FRAC = 0.16
M_UPPER_FRAC = 1.0 - M_STANCE_LEG_FRAC

# MediaPipe world indices (same as mujoco_pipeline)
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28


def vertical_accel_series(
    y: np.ndarray, dt: float, *, max_window: int = 11
) -> np.ndarray:
    """Second time derivative of vertical coordinate (m/s²), SG-smoothed."""
    n = len(y)
    if n < 5:
        return np.zeros(n, dtype=np.float64)
    max_odd = n if n % 2 == 1 else n - 1
    # Tighter max_window preserves stance-phase vertical acceleration peaks vs a 15-tap SG.
    w = min(max_window, max(5, n // 2 * 2 - 1))
    w = min(w, max_odd)
    if w < 5:
        return np.zeros(n, dtype=np.float64)
    return savgol_filter(y, w, polyorder=3, deriv=2, delta=dt, mode="interp").astype(np.float64)


def precompute_two_mass_inputs(processed_landmarks: np.ndarray, dt: float) -> dict[str, np.ndarray]:
    """Hip height and leg COM proxy heights + vertical accelerations."""
    p = processed_landmarks
    n = p.shape[0]
    hip_y = 0.5 * (p[:, L_HIP, 1] + p[:, R_HIP, 1])
    r_leg_y = (p[:, R_HIP, 1] + p[:, R_KNEE, 1] + p[:, R_ANKLE, 1]) / 3.0
    l_leg_y = (p[:, L_HIP, 1] + p[:, L_KNEE, 1] + p[:, L_ANKLE, 1]) / 3.0
    return {
        "a_hip": vertical_accel_series(hip_y, dt),
        "a_r_leg": vertical_accel_series(r_leg_y, dt),
        "a_l_leg": vertical_accel_series(l_leg_y, dt),
    }


def two_mass_vgrf_newtons(
    M_kg: float,
    a_hip: float,
    a_stance_leg: float,
    stance: str,
) -> float:
    """
    stance: "none" | "l" | "r" | "double"
    """
    if stance == "none":
        return 0.0
    if stance == "double":
        return float(M_kg * (a_hip + G))
    m_u = M_kg * M_UPPER_FRAC
    m_s = M_kg * M_STANCE_LEG_FRAC
    return float(m_u * (a_hip + G) + m_s * (a_stance_leg + G))


def stance_label(touch_l: bool, touch_r: bool) -> str:
    if touch_l and touch_r:
        return "double"
    if touch_r:
        return "r"
    if touch_l:
        return "l"
    return "none"


def split_vgrf_to_feet(fy: float, stance: str) -> tuple[float, float]:
    """Returns (fy_left, fy_right) non-negative vertical components."""
    fy = max(0.0, fy)
    if stance == "none":
        return 0.0, 0.0
    if stance == "r":
        return 0.0, fy
    if stance == "l":
        return fy, 0.0
    return 0.5 * fy, 0.5 * fy
