"""
MMPose COCO-17 → image-space **ground / contact hints** for gait and stride timing.

This is complementary to 3D SMPL-based stance in ``two_mass_sprint`` (world Y-up).
It uses **2D ankle image-Y** (OpenCV convention: origin top-left, **y grows downward**).
For a typical side or oblique sprint view, the supporting foot sits **lower in the
frame** (larger ``y / image_height``) than during swing.

**Camera assumptions:** feet move near the bottom of the frame; the envelope of
``max(left_ankle_y, right_ankle_y)`` over time tracks a **local ground band**. This
breaks for upside-down video or strong horizon tilt — use ``joints_coordinate_frame``
and visual QA on ``floor_y_norm`` vs ankles.

Consumers (e.g. stride-kin-solver) can overlay contact curves on the timeline without
running MMPose in the browser when the worker forwards ``metadata.mmpose_gait_2d``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

# COCO body-17 (MMPose HRNet default)
COCO_L_ANKLE = 15
COCO_R_ANKLE = 16


def gait_ground_series_from_coco17(
    keypoints: np.ndarray,
    *,
    image_height_px: float,
    image_width_px: float,
    fps: float,
    kpt_score_thr: float = 0.35,
    floor_window_ratio: float = 0.12,
    clearance_tau_frac: float = 0.018,
) -> dict[str, Any]:
    """
    Parameters
    ----------
    keypoints
        ``(T, 17, 3)`` with ``x, y, score`` in **pixel** coordinates (or already
        normalized x,y in [0,1] if ``image_height_px == 1`` and scores in column 2).
    image_height_px, image_width_px
        Frame size used to normalize y to ``[0, 1]``. If keypoints are already
        normalized, pass ``1.0`` for height and width.
    """
    k = np.asarray(keypoints, dtype=np.float64)
    if k.ndim != 3 or k.shape[1] < 17 or k.shape[2] < 3:
        raise ValueError(f"Expected keypoints (T, 17, 3+); got {k.shape}")
    T = int(k.shape[0])
    H = float(image_height_px) if image_height_px > 0 else 1.0
    W = float(image_width_px) if image_width_px > 0 else 1.0

    yl = k[:, COCO_L_ANKLE, 1] / H
    yr = k[:, COCO_R_ANKLE, 1] / H
    sl = k[:, COCO_L_ANKLE, 2]
    sr = k[:, COCO_R_ANKLE, 2]
    yl = np.where(sl >= kpt_score_thr, yl, np.nan)
    yr = np.where(sr >= kpt_score_thr, yr, np.nan)

    yl = _forward_fill_1d(yl)
    yr = _forward_fill_1d(yr)

    m = np.fmax(yl, yr)
    fill = float(np.nanmedian(m)) if np.any(np.isfinite(m)) else 0.5
    m = np.nan_to_num(m, nan=fill, posinf=fill, neginf=fill)
    m = median_filter(m, size=3, mode="nearest")

    half_w = max(3, min(45, max(int(T * floor_window_ratio) // 2 * 2 + 1, 5)))
    floor_ref = maximum_filter1d(m, size=half_w, mode="nearest")

    amp = float(np.nanmax(m) - np.nanmin(m)) if T > 1 else 0.05
    amp = max(amp, 1e-3)
    tau = float(np.clip(clearance_tau_frac, 0.25 * amp, 0.06))

    clear_l = floor_ref - yl
    clear_r = floor_ref - yr
    hint_l = np.clip(1.0 - np.nan_to_num(clear_l, nan=1.0) / tau, 0.0, 1.0)
    hint_r = np.clip(1.0 - np.nan_to_num(clear_r, nan=1.0) / tau, 0.0, 1.0)

    return {
        "schema_version": 1,
        "coco_layout": "body17",
        "coordinate": "image_y_down_normalized",
        "image_height_px": float(H),
        "image_width_px": float(W),
        "fps": float(fps),
        "kpt_score_thr": float(kpt_score_thr),
        "method": "rolling_max_envelope_on_max(L_ankle,R_ankle)_y_norm",
        "floor_y_norm": [float(x) for x in floor_ref],
        "ankle_l_y_norm": [float(x) if np.isfinite(x) else float("nan") for x in yl],
        "ankle_r_y_norm": [float(x) if np.isfinite(x) else float("nan") for x in yr],
        "clearance_l": [float(x) for x in clear_l],
        "clearance_r": [float(x) for x in clear_r],
        "contact_hint_l": [float(x) for x in hint_l],
        "contact_hint_r": [float(x) for x in hint_r],
    }


def _forward_fill_1d(a: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=np.float64).copy()
    last = np.nan
    for i in range(out.shape[0]):
        if np.isfinite(out[i]):
            last = out[i]
        elif np.isfinite(last):
            out[i] = last
    return out
