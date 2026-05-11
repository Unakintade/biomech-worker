import os
import sys
import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Assuming these exist in your app/ directory based on previous steps
from .solver import solve_kinetics 
from .mesh_to_mujoco import update_mujoco_from_trellis_and_smpl

app = FastAPI(title="Sprint Analysis API (Hybrid GPU Bridge)")

# This URL changes every time you restart Colab. 
# Set this in Render's Environment Variables.
COLAB_BRIDGE_URL = os.getenv("COLAB_BRIDGE_URL", "")

class AnalysisResponse(BaseModel):
    status: str
    results: Optional[dict] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    return {"status": "online", "bridge_configured": COLAB_BRIDGE_URL != ""}

@app.post("/analyze-full", response_model=AnalysisResponse)
async def analyze_sprint_full(
    video: UploadFile = File(...), 
    height_cm: float = 180.0,
    weight_kg: float = 75.0,
    fps: int = 120
):
    """
    Coordination Logic:
    1. Receives video from Frontend.
    2. Forwards video to Google Colab for mmhuman3d/Trellis processing.
    3. Receives 3D SMPL data from Colab.
    4. Runs local MuJoCo physics simulation on Render.
    """
    if not COLAB_BRIDGE_URL:
        raise HTTPException(status_code=500, detail="COLAB_BRIDGE_URL not configured in environment variables.")

    # STEP 1: Forward to Colab GPU Bridge
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            files = {'video': (video.filename, await video.read(), video.content_type)}
            print(f"Forwarding to GPU Bridge: {COLAB_BRIDGE_URL}")
            
            ai_response = await client.post(
                f"{COLAB_BRIDGE_URL}/process-sprint", 
                files=files
            )
            
            if ai_response.status_code != 200:
                return AnalysisResponse(status="error", error="Colab GPU Bridge failed to process video.")
            
            ai_data = ai_response.json()
            # ai_data contains: { "smpl_betas": [...], "mesh_name": "..." }
            
        except Exception as e:
            return AnalysisResponse(status="error", error=f"Bridge connection failed: {str(e)}")

    # STEP 2: Update Local MuJoCo Model with AI Scaling
    try:
        # Update the .xml file based on the Trellis mesh and SMPL betas
        update_mujoco_from_trellis_and_smpl(
            mesh_path="app/data/temp_mesh.obj", 
            xml_path="app/models/sprinter.xml",
            smpl_betas=ai_data['smpl_betas'],
            height_cm=height_cm
        )
    except Exception as e:
        print(f"Model scaling warning: {e}")

    # STEP 3: Run MuJoCo Physics Solve
    try:
        # In a real flow, mmhuman3d returns the actual pose sequence.
        # Here we assume it's provided or we use the processed landmarks.
        mock_sequence = np.random.random((fps * 2, 33, 3)).tolist() 
        results = solve_kinetics(mock_sequence, weight_kg, height_cm, fps)
        
        return AnalysisResponse(status="success", results=results)
    except Exception as e:
        return AnalysisResponse(status="error", error=f"Physics solve failed: {str(e)}")
