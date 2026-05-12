"""
MuJoCo geometric IK (lower body) + mj_forward / mj_inverse for joint torques.
SMPL-24 or MediaPipe-33 world frame: Y up, meters. Gravity matches XML (0 -9.81 0).
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

from .landmark_indices import LandmarkLayout, lower_body_indices, min_joints_for_layout
from .landmark_indices import LowerBodyIndices
from .smpl_joints import (
    HEAD,
    L_ELBOW,
    L_SHOULDER,
    L_WRIST,
    NECK,
    PELVIS,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    SPINE1,
    SPINE3,
)

from .two_mass_sprint import (
    precompute_two_mass_inputs,
    split_vgrf_to_feet,
    sprint_stance_series,
    two_mass_vgrf_newtons,
)

_XML = pathlib.Path(__file__).resolve().parent / "models" / "biped_sprint.xml"

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
    """
    Hinge angles (radians) for ``biped_sprint.xml``.

    Knee / ankle in MJCF use **negative** values for flexion (ranges roughly -2.7..0
    and -1.2..1). The old ``pi - arccos(dot)`` lived in **[0, pi]** and sat **outside**
    those ranges for a straight leg, so MuJoCo clamped the knee and most motion leaked
    into the ankle — matching “only foot/calf” artefacts.
    """
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
    lvec = (ankle - knee) / (np.linalg.norm(ankle - knee) + 1e-9)
    # Internal angle at knee between thigh and shank; flexion = negative of that angle
    # when vectors continue "down" the leg (straight → 0, bent → negative).
    knee_flex = -float(np.arccos(np.clip(float(np.dot(u, lvec)), -1.0, 1.0)))

    v_shin = (knee - ankle) / (np.linalg.norm(knee - ankle) + 1e-9)
    v_foot = (toe - ankle) / (np.linalg.norm(toe - ankle) + 1e-9)
    ankle_flex = -float(np.arccos(np.clip(float(np.dot(v_shin, v_foot)), -1.0, 1.0)))

    return hip_flex, knee_flex, ankle_flex


def _angle_at_joint_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at vertex B, degrees."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    v1 /= n1
    v2 /= n2
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)), -1.0, 1.0))))


def _smpl_upper_body_geometry_deg(wp: np.ndarray) -> dict[str, dict[str, float]]:
    """
    Upper-limb angles from SMPL-24 keypoints only (no MuJoCo shoulder/elbow exists in
    ``biped_spring.xml``). For UI / “activation” proxies; ``torque_nm`` is always 0.
    """
    out: dict[str, dict[str, float | str]] = {}
    try:
        neck = wp[NECK]
        ls, rs = wp[L_SHOULDER], wp[R_SHOULDER]
        le, re = wp[L_ELBOW], wp[R_ELBOW]
        lw, rw = wp[L_WRIST], wp[R_WRIST]
        sp3 = wp[SPINE3]
        hd = wp[HEAD]
        out["neck_tilt"] = {
            "angle_deg": _angle_at_joint_deg(sp3, neck, hd),
            "velocity_rad_s": 0.0,
            "torque_nm": 0.0,
            "estimate": "smpl_keypoint_geometry",
        }
        out["l_shoulder"] = {
            "angle_deg": _angle_at_joint_deg(neck, ls, le),
            "velocity_rad_s": 0.0,
            "torque_nm": 0.0,
            "estimate": "smpl_keypoint_geometry",
        }
        out["r_shoulder"] = {
            "angle_deg": _angle_at_joint_deg(neck, rs, re),
            "velocity_rad_s": 0.0,
            "torque_nm": 0.0,
            "estimate": "smpl_keypoint_geometry",
        }
        out["l_elbow"] = {
            "angle_deg": _angle_at_joint_deg(ls, le, lw),
            "velocity_rad_s": 0.0,
            "torque_nm": 0.0,
            "estimate": "smpl_keypoint_geometry",
        }
        out["r_elbow"] = {
            "angle_deg": _angle_at_joint_deg(rs, re, rw),
            "velocity_rad_s": 0.0,
            "torque_nm": 0.0,
            "estimate": "smpl_keypoint_geometry",
        }
    except (IndexError, TypeError):
        pass
    return out


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


def _yaw_from_landmarks(
    wp: np.ndarray, idx: LowerBodyIndices, layout: LandmarkLayout
) -> float:
    """Facing yaw about +Y from hip line, or pelvis→spine (SMPL) if hips collapse in XZ."""
    hl = wp[idx.l_hip]
    hr = wp[idx.r_hip]
    line = hr - hl
    line_xz = np.array([line[0], 0.0, line[2]], dtype=np.float64)
    ln = float(np.linalg.norm(line_xz))
    if ln > 1e-6:
        return float(np.arctan2(line_xz[0], line_xz[2] + 1e-9))
    if layout == LandmarkLayout.SMPL24:
        f = wp[SPINE1] - wp[PELVIS]
        fxz = np.array([f[0], 0.0, f[2]], dtype=np.float64)
        ln2 = float(np.linalg.norm(fxz))
        if ln2 > 1e-6:
            return float(np.arctan2(fxz[0], fxz[2] + 1e-9))
    return 0.0


def _qpos_from_frame(
    model, wp: np.ndarray, idx: LowerBodyIndices, layout: LandmarkLayout
) -> np.ndarray:
    hl = wp[idx.l_hip]
    hr = wp[idx.r_hip]
    if layout == LandmarkLayout.SMPL24:
        mid = np.array(wp[PELVIS], dtype=np.float64, copy=True)
    else:
        mid = 0.5 * (hl + hr)
    yaw = _yaw_from_landmarks(wp, idx, layout)
    quat = _quat_wxyz_yaw_about_y(yaw)

    rh, rk, ra = _leg_angles(
        hr, wp[idx.r_knee], wp[idx.r_ankle], wp[idx.r_toe], quat, 1.0
    )
    lh, lk, la = _leg_angles(
        hl, wp[idx.l_knee], wp[idx.l_ankle], wp[idx.l_toe], quat, -1.0
    )

    qpos = np.zeros(model.nq, dtype=np.float64)
    qpos[0:3] = mid
    qpos[3:7] = quat
    jvals = [rh, rk, ra, lh, lk, la]
    for name, val in zip(_JOINT_OUT, jvals):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = int(model.jnt_qposadr[jid])
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        if (hi - lo) > 1e-6 and np.isfinite(lo) and np.isfinite(hi):
            val = float(np.clip(val, lo, hi))
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
    mjcf_xml: str | None = None,
    landmark_layout: LandmarkLayout = LandmarkLayout.SMPL24,
) -> list[dict[str, Any]] | None:
    """
    Returns per-frame dicts with joints (angles/vel/torque), com_*, grf_*, vertical_force.
    On failure returns None (caller falls back to landmark-only pipeline).

    Vertical GRF uses a two-mass inertial model (hip + stance-limb accelerations from
    landmarks; stance phase from landmark foot clearance (sprint: no double
    support). Prefer
    ``landmarks_for_vgrf`` (e.g. resampled but not 6 Hz low-pass) so vertical
    accelerations are not overdamped.

    ``mjcf_xml``: optional subject-scaled MJCF string; if omitted, loads bundled
    ``biped_sprint.xml`` from disk.

    ``landmark_layout``: SMPL-24 (Colab / mmhuman3d) vs MediaPipe-33 (stride-kin-solver).
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

    idx = lower_body_indices(landmark_layout)
    min_j = min_joints_for_layout(landmark_layout)
    if processed_landmarks.shape[1] < min_j:
        return None

    try:
        if mjcf_xml is not None:
            model = mujoco.MjModel.from_xml_string(mjcf_xml)
        else:
            model = mujoco.MjModel.from_xml_path(str(_XML))
        data = mujoco.MjData(model)
    except Exception:
        return None

    pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    qpos_seq = np.zeros((n, model.nq), dtype=np.float64)
    for i in range(n):
        q = _qpos_from_frame(model, processed_landmarks[i], idx, landmark_layout)
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
    acc2m = precompute_two_mass_inputs(vgrf_lm, dt, idx)
    stance_labels = sprint_stance_series(vgrf_lm, idx, height_cm=h_cm)

    lb_idx = [idx.l_hip, idx.r_hip, idx.l_knee, idx.r_knee, idx.l_ankle, idx.r_ankle]
    lb_motion_m = float(
        np.max(np.ptp(processed_landmarks[:, lb_idx, :], axis=0))
    )
    static_warn = lb_motion_m < 5e-4

    com_seq = np.zeros((n, 3), dtype=np.float64)
    frames_out: list[dict[str, Any]] = []
    stance_src = (
        "smpl24" if landmark_layout == LandmarkLayout.SMPL24 else "mediapipe33"
    )

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

        joints_out: dict[str, dict[str, float | str]] = {}
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
                "estimate": "mujoco_inverse_dynamics",
            }

        if landmark_layout == LandmarkLayout.SMPL24:
            for k, v in _smpl_upper_body_geometry_deg(processed_landmarks[i]).items():
                if i > 0:
                    prev = float(frames_out[i - 1]["joints"][k]["angle_deg"])
                    v = dict(v)
                    v["velocity_rad_s"] = float(
                        np.radians((v["angle_deg"] - prev) / max(dt, 1e-6))
                    )
                joints_out[k] = v

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

        k3d = processed_landmarks[i].tolist()
        warn_list: list[str] = []
        if static_warn and i == 0:
            warn_list.append(
                "Lower-body keypoints vary <0.5mm over the clip; IK angles may stay ~0. "
                "Verify Colab returns real per-frame SMPL joints (not a repeated template) "
                "and try joints_coordinate_frame: camera_flip_y if upside-down."
            )

        frames_out.append(
            {
                "timestamp": ts,
                "frame_idx": i,
                "keypoints3d": k3d,
                "joints": joints_out,
                "com_position": [float(com_seq[i, 0]), float(com_seq[i, 1]), float(com_seq[i, 2])],
                "com_velocity": [float(vel_com[0]), float(vel_com[1]), float(vel_com[2])],
                "grf_left": [float(fl[0]), float(fl[1]), float(fl[2])],
                "grf_right": [float(fr[0]), float(fr[1]), float(fr[2])],
                "vertical_force": fy_total,
                "two_mass_stance": stance,
                "two_mass_stance_source": stance_src,
                "vgrf_model": "two_mass_sprint_vertical",
                "residual_error": 0.0,
                "warnings": warn_list,
            }
        )

    return frames_out
