"""
MuJoCo geometric IK (lower body) + mj_forward / mj_inverse for joint torques.
MediaPipe world frame: Y up, meters. Gravity matches XML (0 -9.81 0).
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rsci

try:
    import mujoco
except ImportError:
    mujoco = None

from two_mass_sprint import (
    precompute_two_mass_inputs,
    split_vgrf_to_feet,
    sprint_stance_series,
    two_mass_vgrf_newtons,
)

_XML = pathlib.Path(__file__).resolve().parent / "models" / "biped_sprint.xml"

# MediaPipe pose landmarks (world)
R_HIP, R_KNEE, R_ANKLE, R_TOE = 24, 26, 28, 32
L_HIP, L_KNEE, L_ANKLE, L_TOE = 23, 25, 27, 31

_JOINT_OUT = (
    "r_hip_flex",
    "r_knee",
    "r_ankle",
    "l_hip_flex",
    "l_knee",
    "l_ankle",
)


def _quat_wxyz_yaw_about_y(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)


def _mat_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return Rsci.from_quat([x, y, z, w]).as_matrix()


def _leg_angles(
    hip: np.ndarray,
    knee: np.ndarray,
    ankle: np.ndarray,
    toe: np.ndarray,
    quat_wxyz: np.ndarray,
    side_sign: float,
) -> tuple[float, float, float]:
    """Hinge angles (radians) aligned with MJCF (flexion about parent X)."""
    Rw = _mat_from_quat_wxyz(quat_wxyz)
    R_bt = Rw.T

    v_th = knee - hip
    nv = np.linalg.norm(v_th)
    if nv < 1e-8:
        return 0.0, 0.0, 0.0
    v_th /= nv
    loc = R_bt @ v_th
    hip_flex = float(np.arctan2(loc[2], -(loc[1] + 1e-9)) * side_sign)

    u = (knee - hip) / (np.linalg.norm(knee - hip) + 1e-9)
    l = (ankle - knee) / (np.linalg.norm(ankle - knee) + 1e-9)
    knee_flex = float(np.pi - np.arccos(np.clip(float(np.dot(u, l)), -1.0, 1.0)))

    s = (knee - ankle) / (np.linalg.norm(knee - ankle) + 1e-9)
    f = (toe - ankle) / (np.linalg.norm(toe - ankle) + 1e-9)
    ankle_flex = float(np.pi - np.arccos(np.clip(float(np.dot(s, f)), -1.0, 1.0)))
    return hip_flex, knee_flex, ankle_flex


def _elevate_to_clear_floor(model, data, qpos: np.ndarray) -> np.ndarray:
    q = np.array(qpos, copy=True)
    r_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "r_foot_geom")
    l_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "l_foot_geom")
    for _ in range(20):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        y_r = float(data.geom_xpos[r_gid][1])
        y_l = float(data.geom_xpos[l_gid][1])
        ymin = min(y_r, y_l)
        if ymin >= 0.0:
            break
        q[1] += float(-ymin + 0.002)
    return q


def _qpos_from_frame(model, wp: np.ndarray) -> np.ndarray:
    hl = wp[L_HIP]
    hr = wp[R_HIP]
    mid = 0.5 * (hl + hr)
    line = hr - hl
    line_xz = np.array([line[0], 0.0, line[2]], dtype=np.float64)
    ln = float(np.linalg.norm(line_xz))
    yaw = float(np.arctan2(line_xz[0], line_xz[2] + 1e-9)) if ln > 1e-6 else 0.0
    quat = _quat_wxyz_yaw_about_y(yaw)

    rh, rk, ra = _leg_angles(hr, wp[R_KNEE], wp[R_ANKLE], wp[R_TOE], quat, 1.0)
    lh, lk, la = _leg_angles(hl, wp[L_KNEE], wp[L_ANKLE], wp[L_TOE], quat, -1.0)

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:3] = mid
    qpos[3:7] = quat
    jvals = [rh, rk, ra, lh, lk, la]
    for name, val in zip(_JOINT_OUT, jvals):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(model.jnt_qposadr[jid])
        qpos[adr] = val
    return qpos


def _differentiate_qpos(model, q_a: np.ndarray, q_b: np.ndarray, h: float) -> np.ndarray:
    qvel = np.zeros(model.nv, dtype=np.float64)
    mujoco.mj_differentiatePos(model, qvel, h, q_a, q_b)
    return qvel


def run_mujoco_inverse_dynamics(
    processed_landmarks: np.ndarray,
    dt: float,
    weight_kg: float,
    height_cm: float,
    fps: int,
    t_src: np.ndarray,
    landmarks_for_vgrf: np.ndarray | None = None,
) -> list[dict[str, Any]] | None:
    """
    Returns per-frame dicts with joints (angles/vel/torque), com_*, grf_*, vertical_force.
    On failure returns None (caller falls back to landmark-only pipeline).

    Vertical GRF uses a two-mass inertial model (hip + stance-limb accelerations from
    landmarks; stance phase from landmark foot clearance (sprint: no double
    support). Prefer
    ``landmarks_for_vgrf`` (e.g. resampled but not 6 Hz low-pass) so vertical
    accelerations are not overdamped.
    """
    try:
        h_cm = float(height_cm) if height_cm is not None else 0.0
    except (TypeError, ValueError):
        h_cm = 0.0
    _ = fps

    try:
        M_kg = float(weight_kg) if weight_kg is not None and float(weight_kg) > 0 else 75.0
    except (TypeError, ValueError):
        M_kg = 75.0

    if mujoco is None or not _XML.is_file():
        return None

    n = processed_landmarks.shape[0]
    if n < 3:
        return None

    try:
        model = mujoco.MjModel.from_xml_path(str(_XML))
        data = mujoco.MjData(model)
    except Exception:
        return None

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    qpos_seq = np.zeros((n, model.nq), dtype=np.float64)
    for i in range(n):
        q = _qpos_from_frame(model, processed_landmarks[i])
        q = _elevate_to_clear_floor(model, data, q)
        qpos_seq[i] = q

    qvel_seq = np.zeros((n, model.nv), dtype=np.float64)
    for i in range(n):
        if i == 0:
            qvel_seq[i] = _differentiate_qpos(model, qpos_seq[i], qpos_seq[i + 1], dt)
        elif i == n - 1:
            qvel_seq[i] = _differentiate_qpos(model, qpos_seq[i - 1], qpos_seq[i], dt)
        else:
            qvel_seq[i] = _differentiate_qpos(model, qpos_seq[i - 1], qpos_seq[i + 1], 2.0 * dt)

    qacc_seq = np.zeros((n, model.nv), dtype=np.float64)
    for i in range(n):
        if i == 0 or i == n - 1:
            qacc_seq[i] = 0.0
        else:
            qacc_seq[i] = (qvel_seq[i + 1] - qvel_seq[i - 1]) / (2.0 * dt)

    vgrf_lm = (
        landmarks_for_vgrf
        if landmarks_for_vgrf is not None
        and landmarks_for_vgrf.shape == processed_landmarks.shape
        else processed_landmarks
    )
    acc2m = precompute_two_mass_inputs(vgrf_lm, dt)
    stance_labels = sprint_stance_series(vgrf_lm, height_cm=h_cm)

    com_seq = np.zeros((n, 3), dtype=np.float64)
    frames_out: list[dict[str, Any]] = []

    for i in range(n):
        data.qpos[:] = qpos_seq[i]
        data.qvel[:] = qvel_seq[i]
        mujoco.mj_forward(model, data)
        try:
            com_seq[i] = np.array(data.subtree_com[pelvis_bid], dtype=np.float64)
        except (AttributeError, IndexError, TypeError):
            com_seq[i] = np.array(data.xipos[pelvis_bid], dtype=np.float64)

        data.qacc[:] = qacc_seq[i]
        mujoco.mj_inverse(model, data)
        tau = np.array(data.qfrc_inverse, dtype=np.float64)

        joints_out: dict[str, dict[str, float]] = {}
        for jname in _JOINT_OUT:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = int(model.jnt_qposadr[jid])
            vadr = int(model.jnt_dofadr[jid])
            angle_deg = float(np.degrees(qpos_seq[i, qadr]))
            vel_rs = float(qvel_seq[i, vadr])
            torque = float(tau[vadr])
            joints_out[jname] = {
                "angle_deg": angle_deg,
                "velocity_rad_s": vel_rs,
                "torque_nm": torque,
            }

        stance = str(stance_labels[i])
        if stance == "r":
            a_st = float(acc2m["a_r_leg"][i])
        elif stance == "l":
            a_st = float(acc2m["a_l_leg"][i])
        else:
            a_st = 0.0

        fy_total = two_mass_vgrf_newtons(M_kg, float(acc2m["a_hip"][i]), a_st, stance)
        fy_l, fy_r = split_vgrf_to_feet(fy_total, stance)
        fl = np.array([0.0, fy_l, 0.0], dtype=np.float64)
        fr = np.array([0.0, fy_r, 0.0], dtype=np.float64)
        ts = float(t_src[i]) if i < len(t_src) else float(i) * dt

        vel_com = np.zeros(3, dtype=np.float64)
        if i > 0 and i < n - 1:
            vel_com = (com_seq[i + 1] - com_seq[i - 1]) / (2.0 * dt)
        elif i > 0:
            vel_com = (com_seq[i] - com_seq[i - 1]) / dt
        elif n > 1:
            vel_com = (com_seq[i + 1] - com_seq[i]) / dt

        frames_out.append(
            {
                "timestamp": ts,
                "frame_idx": i,
                "joints": joints_out,
                "com_position": [float(com_seq[i, 0]), float(com_seq[i, 1]), float(com_seq[i, 2])],
                "com_velocity": [float(vel_com[0]), float(vel_com[1]), float(vel_com[2])],
                "grf_left": [float(fl[0]), float(fl[1]), float(fl[2])],
                "grf_right": [float(fr[0]), float(fr[1]), float(fr[2])],
                "vertical_force": fy_total,
                "two_mass_stance": stance,
                "two_mass_stance_source": "landmark_sprint",
                "vgrf_model": "two_mass_sprint_vertical",
                "residual_error": 0.0,
                "warnings": [],
            }
        )

    return frames_out
