import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    frameIdx: int
    timestamp: float
    # worldPositions is [33 landmarks][x, y, z, visibility]
    worldPositions: List[List[float]] 

class AnalysisRequest(BaseModel):
    # Renamed to 'landmarks' to match your frontend JSON
    landmarks: List[FrameData]
    weight_kg: float
    height_cm: float
    fps: int

# Import solver after defining models
from solver import solve_kinetics

@app.get("/health")
def health():
    return {"status": "online", "engine": "MuJoCo 3.x"}

@app.post("/analyze")
async def analyze_sprint(request: AnalysisRequest):
    try:
        # 3. CONVERT DATA FOR MATH
        # We need a 3D array: [frames, landmarks, 3-coords]
        processed_data = []
        for frame in request.landmarks:
            # We take only x, y, z (the first 3 numbers) from worldPositions
            coords = [p[:3] for p in frame.worldPositions]
            processed_data.append(coords)
            
        data_array = np.array(processed_data)
        
        # 4. RUN PHYSICS
        results = solve_kinetics(
            data_array, 
            request.weight_kg, 
            request.height_cm, 
            request.fps
        )
        
        return results
    except Exception as e:
        # Log the error for Render logs
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
