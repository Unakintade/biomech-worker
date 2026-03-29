"""
Two-mass vertical model for sprinting vGRF (sagittal / vertical dynamics).

Treats the runner as:
  - m_u: upper lump (HAT + pelvis + swing-side mass lumped with trunk proxy) — motion from hip height
  - m_s: stance-limb lump — vertical motion from the stance leg COM proxy (hip–knee–ankle mean)

During single support (one foot on ground — stance from landmarks, not MuJoCo):
    F_y ≈ m_u * (a_hip + g) + m_s * (a_stance_leg + g)

Sprint stance never uses double support: if both feet are near the local floor
estimate, the lower foot (smaller world Y) is stance.

Flight (no foot near local floor): F_y = 0.

World landmarks: vertical is axis 1 (Y up). MuJoCo foot–floor contact is not
used for stance because IK + per-frame floor elevation forces both feet onto
the plane and creates spurious double support.

Mass fractions follow typical segment tables (~one leg ≈ 16% body mass, remainder ≈ upper lump).
Y-up world frame (MediaPipe / current MJCF).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter, minimum_filter1d
from scipy.signal import savgol_filter

G = 9.81

# One leg (thigh+shank+foot) ~16% M (De Leva–style); rest assigned to upper lump for this vertical split
M_STANCE_LEG_FRAC = 0.16
M_UPPER_FRAC = 1.0 - M_STANCE_LEG_FRAC

# MediaPipe world indices (same as mujoco_pipeline)
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_TOE, R_TOE = 31, 32


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


def sprint_stance_series(
    p: np.ndarray,
    *,
    height_cm: float = 0.0,
    contact_margin_m: float = 0.036,
) -> np.ndarray:
    """
    Per-frame stance for sprinting: 'none' | 'l' | 'r' only.

    Uses lowest world-Y point per foot (ankle vs toe). Local floor reference is a
    temporal minimum of min(left, right) so camera / global offset matters less
    than each foot's clearance relative to the stride trough. Never emits
    ``double`` — ties go to the geometrically lower foot, then previous label.
    """
    n = int(p.shape[0])
    if n < 3:
        return np.array(["none"] * max(n, 0), dtype=object)

    l_min = np.minimum(p[:, L_ANKLE, 1], p[:, L_TOE, 1])
    r_min = np.minimum(p[:, R_ANKLE, 1], p[:, R_TOE, 1])
    l_min = median_filter(l_min.astype(np.float64), size=3, mode="nearest")
    r_min = median_filter(r_min.astype(np.float64), size=3, mode="nearest")

    m = np.minimum(l_min, r_min)
    half_w = max(5, min(31, max(n // 6, 5)))
    floor_ref = minimum_filter1d(m, size=2 * half_w + 1, mode="nearest")
    apex_ref = maximum_filter1d(m, size=2 * half_w + 1, mode="nearest")
    stride_amp = np.maximum(apex_ref - floor_ref, 1e-4)

    scale = (float(height_cm) / 175.0) if height_cm and float(height_cm) > 120 else 1.0
    tau = float(contact_margin_m) * scale
    # Near local trough: scale with stride height, but keep a floor so flat/noisy clips still classify.
    tau_eff = np.minimum(
        tau,
        np.maximum(0.22 * stride_amp, 0.72 * tau),
    )

    out = np.empty(n, dtype=object)
    prev = "none"
    for i in range(n):
        fr = float(floor_ref[i])
        cl = float(l_min[i]) <= fr + float(tau_eff[i])
        cr = float(r_min[i]) <= fr + float(tau_eff[i])
        if cl and cr:
            if float(l_min[i]) < float(r_min[i]) - 1e-5:
                prev = "l"
            elif float(r_min[i]) < float(l_min[i]) - 1e-5:
                prev = "r"
            else:
                prev = (
                    prev
                    if prev in ("l", "r")
                    else ("l" if float(l_min[i]) <= float(r_min[i]) else "r")
                )
        elif cl:
            prev = "l"
        elif cr:
            prev = "r"
        else:
            prev = "none"
        out[i] = prev
    return out


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
