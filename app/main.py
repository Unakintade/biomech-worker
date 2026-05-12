import os
import sys
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict, ValidationError
from typing import List, Optional
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from .bridge_schema import parse_colab_result
from .landmark_indices import LandmarkLayout, lower_body_indices
from .mjcf_smpl_scale import build_scaled_biped_mjcf_xml
from .mujoco_pipeline import run_mujoco_inverse_dynamics


# 1. DEFINE APP FIRST (Fixes NameError)
app = FastAPI(title="Sprint Analysis API (Colab mmhuman3d + MuJoCo)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

COLAB_BRIDGE_URL = os.getenv("COLAB_BRIDGE_URL", "")
BRIDGE_TIMEOUT_SEC = float(os.getenv("BRIDGE_TIMEOUT_SEC", "300"))


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
    weight_kg: float 
    height_cm: float | None = None
    anthropometry: dict[str, float] | None = None


@app.get("/")
async def root():
    return {"status": "online", "bridge_configured": COLAB_BRIDGE_URL != ""}

@app.get("/health")
async def health():
    """Used by stride-kin-solver MuJoCo panel (GET /health)."""
    return {"status": "ok"}

@app.post("/analyze-full", response_model=AnalysisResponse)
async def analyze_sprint_full(
    video: UploadFile = File(...), 
    height_cm: float = 180.0,
    weight_kg: float = 75.0,
    fps: float = 120.0,
):
    """
    Coordination Logic:
    1. Receives video from Frontend.
    2. Forwards video to Google Colab for mmhuman3d/Trellis processing.
    3. Receives 3D SMPL data from Colab.
    4. Runs local MuJoCo physics simulation on Render.
    """
    if not COLAB_BRIDGE_URL:
        raise HTTPException(
            status_code=500,
            detail="COLAB_BRIDGE_URL not configured in environment variables.",
        )

    # STEP 1: Forward to Colab GPU Bridge
    base = COLAB_BRIDGE_URL.rstrip("/")

    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SEC) as client:
        try:
            content = await video.read()
            files = {"video": (video.filename, content, video.content_type)}
            ai_response = await client.post(f"{base}/process-sprint", files=files)
        except httpx.HTTPError as e:
            return AnalysisResponse(
                status="error",
                error=f"Bridge connection failed: {e!s}",
            )
            
            
    if ai_response.status_code != 200:
        return AnalysisResponse(
            status="error",
            error=f"Colab bridge HTTP {ai_response.status_code}: {ai_response.text[:500]}",
        )

    try:
        colab = parse_colab_result(ai_response.json())
    except ValidationError as e:
        return AnalysisResponse(
            status="error",
            error=f"Invalid bridge JSON (expected SMPL fields): {e!s}",
        )

    fps_use = float(colab.fps) if colab.fps is not None else float(fps)
    fps_use = max(fps_use, 1e-3)
    dt = 1.0 / fps_use

    joints = colab.joints_numpy_mujoco()
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

    idx = lower_body_indices(LandmarkLayout.SMPL24)
    lb_idx = [idx.l_hip, idx.r_hip, idx.l_knee, idx.r_knee, idx.l_ankle, idx.r_ankle]
    lower_body_peak_motion_m = float(np.max(np.ptp(joints[:, lb_idx, :], axis=0)))

    return AnalysisResponse(
        status="success",
        results={
            "metadata": {
                "joint_source": "smpl24",
                "fps": fps_use,
                "num_frames": len(frames),
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "joints_coordinate_frame": colab.joints_coordinate_frame,
                "lower_body_peak_motion_m": lower_body_peak_motion_m,
                "keypoints3d_layout": "smpl24_per_frame",
            },
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
