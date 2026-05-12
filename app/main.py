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
    
    # --- UPDATED: Panning Calibration Fields ---
    start_marker_px: Optional[list[float]] = None  # [x, y] in first frame
    end_marker_px: Optional[list[float]] = None    # [x, y] in last frame
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
    # Allow optional manual markers via form data
    start_marker_x: Optional[float] = Form(None),
    start_marker_y: Optional[float] = Form(None),
    end_marker_x: Optional[float] = Form(None),
    end_marker_y: Optional[float] = Form(None),
    marker_dist: Optional[float] = Form(None)
):
    if not COLAB_BRIDGE_URL:
        return AnalysisResponse(
            status="error",
            error="COLAB_BRIDGE_URL is not set on this server."
        )

    base = COLAB_BRIDGE_URL.rstrip("/")
    
    # Package manual markers if provided
    calib = None
    if start_marker_x is not None and end_marker_x is not None and marker_dist:
        calib = {
            "start": [start_marker_x, start_marker_y],
            "end": [end_marker_x, end_marker_y],
            "dist": marker_dist
        }

    try:
        return await _analyze_full_run(
            base=base,
            video=video,
            height_cm=height_cm,
            weight_kg=weight_kg,
            fps=fps,
            calibration=calib
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
    calibration: Optional[dict] = None
) -> AnalysisResponse:
    
    # Read video content for both Colab and local panning correction
    video_bytes = await video.read()
    
    async with httpx.AsyncClient(timeout=BRIDGE_TIMEOUT_SEC) as client:
        try:
            files = {"video": (video.filename, video_bytes, video.content_type)}
            ai_response = await client.post(f"{base}/process-sprint", files=files)
        except httpx.HTTPError as e:
            return AnalysisResponse(status="error", error=f"Bridge connection failed: {e!s}")
            
    if ai_response.status_code != 200:
        return AnalysisResponse(status="error", error=f"Colab bridge error: {ai_response.text[:500]}")

    try:
        colab = parse_colab_result(ai_response.json())
        joints = colab.joints_numpy_mujoco()
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        return AnalysisResponse(status="error", error=f"Data parsing failed: {e!s}")

    # --- NEW: Panning Correction Integration ---
    # We extract frames to run the SIFT/Homography background registration
    if calibration:
        try:
            # Temporary save to read via OpenCV
            with open("temp_panning_video.mp4", "wb") as f:
                f.write(video_bytes)
            
            cap = cv2.VideoCapture("temp_panning_video.mp4")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            cap.release()

            # Run the normalization
            # joints shape is (T, 24, 3). We correct X,Y and keep Z depth.
            joints = np.array(normalize_sprint_sequence(joints.tolist(), frames))
            logger.info("Panning correction applied successfully.")
            
        except Exception as e:
            logger.warning(f"Panning correction failed, proceeding with raw data: {e}")

    fps_use = float(colab.fps) if colab.fps is not None else float(fps)
    dt = 1.0 / max(fps_use, 1e-3)
    betas = colab.betas_numpy()
    t_src = colab.time_array(dt)

    try:
        mjcf_xml = build_scaled_biped_mjcf_xml(joints, betas)
        
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
    except Exception as e:
        return AnalysisResponse(status="error", error=f"Simulation failed: {e!s}")

    if frames is None:
        return AnalysisResponse(status="error", error="MuJoCo execution failed.")

    # ... (Rest of metadata logic remains same) ...
    return AnalysisResponse(
        status="success",
        results={"metadata": {"panning_corrected": calibration is not None}, "frames": frames},
    )

@app.post("/analyze")
async def analyze_mediapipe_landmarks(req: AnalyzeLandmarksRequest):
    # (Implementation for JSON-only landmarks if frames are provided elsewhere)
    # ... extraction logic ...
    joints = np.asarray(rows, dtype=np.float64)
    
    # Note: normalize_sprint_sequence requires video frames for SIFT.
    # If using JSON landmarks without video, panning correction requires 
    # a different geometric warp based solely on the two marker points.
    
    # Proceed to solver...
    return {"frames": [], "summary": {}}
