import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from typing import List, Optional
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Ensure the 'app' folder is in the path so we can find solver.py
sys.path.append(os.path.dirname(__file__))

# 1. DEFINE APP FIRST (Fixes NameError)
app = FastAPI(title="Sprinter Biomechanics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. MATCH FRONTEND DATA STRUCTURE (Fixes 422 Error)

class FrameData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frameIdx: int
    timestamp: float
    worldPositions: List[List[float]]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    landmarks: List[FrameData]
    weight_kg: float
    height_cm: Optional[float] = None
    fps: int


from solver import solve_kinetics


@app.get("/health")
def health():
    return {"status": "online", "engine": "landmark-kinetics-v2 (MediaPipe world + SG)"}


@app.post("/analyze")
async def analyze_sprint(request: AnalysisRequest):
    try:
        processed_data = []
        timestamps = []
        for frame in request.landmarks:
            coords = [p[:3] for p in frame.worldPositions]
            processed_data.append(coords)
            timestamps.append(float(frame.timestamp))

        data_array = np.array(processed_data)
        height_cm = request.height_cm if request.height_cm is not None else 0.0

        results = solve_kinetics(
            data_array,
            request.weight_kg,
            height_cm,
            request.fps,
            timestamps,
        )
        return results
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
