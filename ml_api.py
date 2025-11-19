# ml_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import os
from datetime import datetime
app = FastAPI()

# Import your ML pipeline
try:
    from glucose_pipeline import GlucosePredictor
except ImportError:
    # For Railway deployment
    import sys
    sys.path.append('/app')
    from glucose_pipeline import GlucosePredictor

app = FastAPI(
    title="Glucose Prediction API",
    description="ML API for predicting post-meal glucose levels",
    version="1.0.0"
)

# CORS middleware - allow your React app domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://your-react-app.vercel.app",  # Your deployed React app
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = GlucosePredictor()

# Request/Response models
class PredictionRequest(BaseModel):
    patient_id: str
    dosage: float
    cbg: float
    carbohydrates_estimation: float
    meal_type: str
    rice_cups: Optional[float] = 1.0
    time: Optional[str] = None

class PredictionResponse(BaseModel):
    patient_id: str
    predicted_glucose: float
    prediction_timestamp: str
    model_confidence: str
    expected_accuracy: str

class TrainingResponse(BaseModel):
    status: str
    patient_id: str
    training_records: Optional[int] = None
    metrics: Optional[Dict] = None
    message: Optional[str] = None

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "Glucose Prediction API is running!",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_glucose(request: PredictionRequest):
    """Make glucose prediction"""
    try:
        input_data = {
            'time': request.time or datetime.now().isoformat(),
            'dosage': request.dosage,
            'cbg': request.cbg,
            'cbg_pre_meal': request.cbg,
            'carbohydrates_estimation': request.carbohydrates_estimation,
            'rice_cups': request.rice_cups,
            'meal_type': request.meal_type
        }
        
        result = predictor.predict(request.patient_id, input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/{patient_id}", response_model=TrainingResponse)
async def train_model(patient_id: str):
    """Train model for specific patient"""
    try:
        result = predictor.train_patient_model(patient_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{patient_id}")
async def get_model_status(patient_id: str):
    """Get model status"""
    try:
        return predictor.get_status(patient_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# For Railway deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/health")
def health():
    return {"status": "ok"}
