import os
import json
import logging
import sys
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import List, Optional
import numpy as np
import cv2  # Required for frame extraction
from fastapi.middleware.cors import CORSMiddleware

from .bridge_schema import parse_colab_result
from .landmark_indices import LandmarkLayout, lower_body_indices
from .mjcf_smpl_scale import build_scaled_biped_mjcf_xml
from .mujoco_pipeline import run_mujoco_inverse_dynamics
# Import the new panning utility
from .panning_utils import normalize_sprint_sequence

logger = logging.getLogger(__name__)

app = FastAPI(title="Sprint Analysis API (Colab mmhuman3d + MuJoCo)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

COLAB_BRIDGE_URL = os.getenv("COLAB_BRIDGE_URL", "")
BRIDGE_TIMEOUT_SEC = float(os.getenv("BRIDGE_TIMEOUT_SEC", "300"))


def _colab_tunnel_headers(base_url: str) -> dict[str, str]:
    if "ngrok" in base_url.lower():
        return {"ngrok-skip-browser-warning": "true"}
    return {}


def _format_bridge_http_error(status_code: int, body: str) -> str:
    snippet = (body or "").strip()
    if snippet.startswith("<!DOCTYPE") or snippet.startswith("<html"):
        snippet = "ngrok/HTML error page (upstream not reachable or tunnel misconfigured)"
    else:
        snippet = snippet[:400]
    hint = ""
    if status_code in (502, 503, 504):
        hint = (
            " Usually: Colab cell stopped, FastAPI not listening on 0.0.0.0, wrong port in ngrok, "
            "or GPU runtime disconnected. Restart the Colab server cell and confirm the bridge URL "
            "opens /docs in a browser."
        )
    return f"Colab bridge HTTP {status_code}: {snippet}.{hint}"


class AnalysisResponse(BaseModel):
    status: str
    results: dict | None = None
    error: str | None = None


class LandmarkFrameIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frameIdx: int
    timestamp: float
    worldPositions: list[list[float]]
    visibility: list[float] | None = None


class AnalyzeLandmarksRequest(BaseModel):
    """Payload from stride-kin-solver (MediaPipe world landmarks)."""

    model_config = ConfigDict(extra="ignore")

    landmarks: list[LandmarkFrameIn]
    fps: float
    weight_kg: float = 75.0
    height_cm: float | None = None
    anthropometry: dict[str, float] | None = None

    start_marker_px: Optional[list[float]] = None
    end_marker_px: Optional[list[float]] = None
    marker_distance_m: Optional[float] = None


@app.get("/")
async def root():
    return {"status": "online", "bridge_configured": COLAB_BRIDGE_URL != ""}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze-full", response_model=AnalysisResponse)
async def analyze_sprint_full(
    video: UploadFile = File(...),
    height_cm: float = 180.0,
    weight_kg: float = 75.0,
    fps: float = 120.0,
    start_marker_x: Optional[float] = Form(None),
    start_marker_y: Optional[float] = Form(None),
    end_marker_x: Optional[float] = Form(None),
    end_marker_y: Optional[float] = Form(None),
    marker_dist: Optional[float] = Form(None),
):
    if not COLAB_BRIDGE_URL:
        return AnalysisResponse(
            status="error",
            error=(
                "COLAB_BRIDGE_URL is not set on this server. "
                "In Render: Environment → add COLAB_BRIDGE_URL to your Colab/ngrok base URL (no trailing slash)."
            ),
        )

    base = COLAB_BRIDGE_URL.rstrip("/")

    calib: dict | None = None
    if (
        start_marker_x is not None
        and start_marker_y is not None
        and end_marker_x is not None
        and end_marker_y is not None
        and marker_dist is not None
    ):
        calib = {
            "start": [float(start_marker_x), float(start_marker_y)],
            "end": [float(end_marker_x), float(end_marker_y)],
            "dist": float(marker_dist),
        }

    try:
        return await _analyze_full_run(
            base=base,
            video=video,
            height_cm=height_cm,
            weight_kg=weight_kg,
            fps=fps,
            calibration=calib,
        )
    except Exception as e:
        logger.exception("analyze-full failed")
        return AnalysisResponse(
            status="error",
            error=f"Unexpected server error: {e!s}",
        )


async def _analyze_full_run(
    *,
    base: str,
    video: UploadFile,
    height_cm: float,
    weight_kg: float,
    fps: float,
    calibration: dict | None = None,
) -> AnalysisResponse:
    video_bytes = await video.read()

    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SEC) as client:
        try:
            files = {"video": (video.filename, video_bytes, video.content_type)}
            ai_response = await client.post(
                f"{base}/process-sprint",
                files=files,
                headers=_colab_tunnel_headers(base),
            )
        except httpx.HTTPError as e:
            return AnalysisResponse(
                status="error",
                error=f"Bridge connection failed: {e!s}",
            )

    if ai_response.status_code != 200:
        return AnalysisResponse(
            status="error",
            error=_format_bridge_http_error(
                ai_response.status_code, ai_response.text
            ),
        )

    try:
        raw = ai_response.json()
    except json.JSONDecodeError:
        return AnalysisResponse(
            status="error",
            error="Colab bridge returned non-JSON (check tunnel URL and ngrok / Colab logs).",
        )

    vertices_raw = raw.get("vertices")
    smpl_faces_raw = raw.get("smpl_faces")

    try:
        colab = parse_colab_result(raw)
    except ValidationError as e:
        return AnalysisResponse(
            status="error",
            error=f"Invalid bridge JSON (expected SMPL fields): {e!s}",
        )

    try:
        joints = colab.joints_numpy_mujoco()
    except ValueError as e:
        return AnalysisResponse(
            status="error",
            error=f"joints_world shape/content invalid (need T×24×3 in meters): {e!s}",
        )

    if calibration:
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                tf.write(video_bytes)
                tmp_path = tf.name

            cap = cv2.VideoCapture(tmp_path)
            frames_bgr: list[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
            cap.release()

            joints = np.asarray(
                normalize_sprint_sequence(joints.tolist(), frames_bgr),
                dtype=np.float64,
            )
            logger.info("Panning correction path completed (see panning_utils for algorithm).")
        except Exception as e:
            logger.warning("Panning correction failed, proceeding with raw joints: %s", e)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    fps_use = float(colab.fps) if colab.fps is not None else float(fps)
    fps_use = max(fps_use, 1e-3)
    dt = 1.0 / fps_use
    betas = colab.betas_numpy()
    t_src = colab.time_array(dt)

    try:
        mjcf_xml = build_scaled_biped_mjcf_xml(joints, betas)
    except Exception as e:
        return AnalysisResponse(
            status="error",
            error=f"MJCF scaling failed: {e!s}",
        )

    frames = run_mujoco_inverse_dynamics(
        joints,
        dt,
        weight_kg,
        height_cm,
        int(round(fps_use)),
        t_src,
        landmarks_for_vgrf=None,
        mjcf_xml=mjcf_xml,
    )

    if frames is None:
        return AnalysisResponse(
            status="error",
            error="MuJoCo inverse dynamics failed (check logs / MJCF validity).",
        )

    T = len(frames)
    if isinstance(vertices_raw, list) and len(vertices_raw) == T:
        for i, fr in enumerate(frames):
            if isinstance(fr, dict):
                v = vertices_raw[i]
                if hasattr(v, "tolist"):
                    fr["vertices"] = v.tolist()
                else:
                    fr["vertices"] = v

    idx = lower_body_indices(LandmarkLayout.SMPL24)
    lb_idx = [idx.l_hip, idx.r_hip, idx.l_knee, idx.r_knee, idx.l_ankle, idx.r_ankle]
    try:
        lower_body_peak_motion_m = float(np.max(np.ptp(joints[:, lb_idx, :], axis=0)))
    except Exception:
        lower_body_peak_motion_m = 0.0

    jcf = colab.joints_coordinate_frame
    if hasattr(jcf, "value"):
        jcf = jcf.value

    bridge_meta = dict(colab.metadata) if colab.metadata else {}
    meta_out: dict = {**bridge_meta}
    meta_out.update(
        {
            "joint_source": "smpl24",
            "fps": fps_use,
            "num_frames": len(frames),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "joints_coordinate_frame": str(jcf),
            "lower_body_peak_motion_m": lower_body_peak_motion_m,
            "keypoints3d_layout": "smpl24_per_frame",
            "panning_corrected": calibration is not None,
        }
    )

    if smpl_faces_raw is not None:
        if hasattr(smpl_faces_raw, "tolist"):
            meta_out["smpl_faces"] = smpl_faces_raw.tolist()
        else:
            meta_out["smpl_faces"] = smpl_faces_raw

    return AnalysisResponse(
        status="success",
        results={
            "metadata": meta_out,
            "frames": frames,
        },
    )


@app.post("/analyze")
async def analyze_mediapipe_landmarks(req: AnalyzeLandmarksRequest):
    """
    stride-kin-solver compatibility: JSON landmarks (33 world points per frame) → MuJoCo.
    Response shape matches ``mujocoApi.normaliseResponse`` (top-level ``frames`` + ``summary``).
    """
    if len(req.landmarks) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 frames.")

    rows: list[list[list[float]]] = []
    for fr in req.landmarks:
        wp = fr.worldPositions
        if len(wp) < 33:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 33 world landmarks per frame; got {len(wp)}.",
            )
        rows.append([list(map(float, p[:3])) for p in wp[:33]])

    joints = np.asarray(rows, dtype=np.float64)
    fps_use = max(float(req.fps), 1e-3)
    dt = 1.0 / fps_use
    t_src = np.asarray([fr.timestamp for fr in req.landmarks], dtype=np.float64)
    if t_src.shape[0] != joints.shape[0]:
        t_src = np.arange(joints.shape[0], dtype=np.float64) * dt

    height_cm = float(req.height_cm) if req.height_cm is not None else 0.0

    frames = run_mujoco_inverse_dynamics(
        joints,
        dt,
        req.weight_kg,
        height_cm,
        int(round(fps_use)),
        t_src,
        landmarks_for_vgrf=None,
        mjcf_xml=None,
        landmark_layout=LandmarkLayout.MEDIAPIPE33,
    )
    if frames is None:
        raise HTTPException(
            status_code=500,
            detail="MuJoCo inverse dynamics failed (check landmark quality / FPS).",
        )

    n_warn = sum(len(f.get("warnings") or []) for f in frames)
    return {
        "frames": frames,
        "summary": {
            "total_frames": len(frames),
            "solve_time_s": 0.0,
            "mean_residual_m": 0.0,
            "max_residual_m": 0.0,
            "total_warnings": n_warn,
            "fps": fps_use,
        },
    }
