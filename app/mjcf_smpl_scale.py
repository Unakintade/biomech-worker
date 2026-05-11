"""
Build a subject-scaled biped MJCF from SMPL joints + betas (no on-disk mutation).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .smpl_joints import L_ANKLE, L_FOOT, L_HIP, L_KNEE, R_ANKLE, R_FOOT, R_HIP, R_KNEE

_TEMPLATE = Path(__file__).resolve().parent / "models" / "biped_sprint.xml"

# Default capsule radii in template (m)
_THIGH_R = 0.055
_SHANK_R = 0.048
_FOOT_SPHERE_R = 0.055


def _mid_frame_idx(n: int) -> int:
    return int(np.clip(n // 2, 0, max(0, n - 1)))


def smpl_leg_segment_lengths_m(joints: np.ndarray, frame_idx: int | None = None) -> tuple[float, float]:
    """Mean L/R thigh and shank bone lengths (m) from SMPL joint positions."""
    t = _mid_frame_idx(len(joints)) if frame_idx is None else int(np.clip(frame_idx, 0, len(joints) - 1))
    j = joints[t]
    lt = float(np.linalg.norm(j[L_KNEE] - j[L_HIP]))
    rt = float(np.linalg.norm(j[R_KNEE] - j[R_HIP]))
    ls = float(np.linalg.norm(j[L_ANKLE] - j[L_KNEE]))
    rs = float(np.linalg.norm(j[R_ANKLE] - j[R_KNEE]))
    thigh = max(0.05, 0.5 * (lt + rt))
    shank = max(0.05, 0.5 * (ls + rs))
    return thigh, shank


def _foot_radius_scale(joints: np.ndarray, frame_idx: int | None = None) -> float:
    t = _mid_frame_idx(len(joints)) if frame_idx is None else int(np.clip(frame_idx, 0, len(joints) - 1))
    j = joints[t]
    lf = float(np.linalg.norm(j[L_FOOT] - j[L_ANKLE]))
    rf = float(np.linalg.norm(j[R_FOOT] - j[R_ANKLE]))
    foot = 0.5 * (lf + rf)
    ref = 0.19
    return float(np.clip(foot / ref, 0.7, 1.4))


def _thickness_mult(betas: np.ndarray) -> float:
    if betas.size < 2:
        return 1.0
    m = 1.0 + float(betas[1]) * 0.1
    return float(np.clip(m, 0.85, 1.25))


def build_scaled_biped_mjcf_xml(joints: np.ndarray, betas: np.ndarray) -> str:
    """
    Load biped_sprint.xml template, apply SMPL-derived segment lengths and beta thickness,
    return MJCF as a string for mujoco.MjModel.from_xml_string.
    """
    thigh, shank = smpl_leg_segment_lengths_m(joints)
    foot_scale = _foot_radius_scale(joints)
    thick = _thickness_mult(np.asarray(betas, dtype=np.float64).ravel())

    tree = ET.parse(_TEMPLATE)
    root = tree.getroot()

    def set_body_y(name: str, y_neg: float) -> None:
        el = next((b for b in root.iter("body") if b.get("name") == name), None)
        if el is None:
            raise RuntimeError(f"body '{name}' not found in biped template")
        el.set("pos", f"0 {-abs(y_neg):.6f} 0")

    def set_geom_fromto(name: str, length: float) -> None:
        el = next((g for g in root.iter("geom") if g.get("name") == name), None)
        if el is None:
            raise RuntimeError(f"geom '{name}' not found in biped template")
        el.set("fromto", f"0 0 0 0 {-abs(length):.6f} 0")

    def set_geom_size(name: str, radius: float) -> None:
        el = next((g for g in root.iter("geom") if g.get("name") == name), None)
        if el is None:
            raise RuntimeError(f"geom '{name}' not found in biped template")
        el.set("size", f"{radius:.6f}")

    for side in ("r", "l"):
        set_body_y(f"{side}_shank", thigh)
        set_body_y(f"{side}_foot", shank)
        set_geom_fromto(f"{side}_thigh_geom", thigh)
        set_geom_fromto(f"{side}_shank_geom", shank)
        set_geom_size(f"{side}_thigh_geom", _THIGH_R * thick)
        set_geom_size(f"{side}_shank_geom", _SHANK_R * thick)
        set_geom_size(f"{side}_foot_geom", _FOOT_SPHERE_R * foot_scale * thick)

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    return ET.tostring(root, encoding="unicode")
