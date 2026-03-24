from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from .solver import solve_kinetics
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sprinter Biomechanics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float

class AnalysisRequest(BaseModel):
    landmarks_sequence: List[List[Landmark]]  # Nested list: [frame][landmark_id]
    weight_kg: float
    height_cm: float
    fps: int

@app.get("/health")
def health():
    return {"status": "online", "engine": "MuJoCo 3.x"}

@app.post("/analyze")
async def analyze_sprint(request: AnalysisRequest):
    try:
        # Convert Pydantic models to Numpy arrays for processing
        data_array = np.array([
            [[lm.x, lm.y, lm.z] for lm in frame] 
            for frame in request.landmarks_sequence
        ])
        
        # Call the MuJoCo Solver (Inverse Dynamics)
        results = solve_kinetics(
            data_array, 
            request.weight_kg, 
            request.height_cm, 
            request.fps
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
