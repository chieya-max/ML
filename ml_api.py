# ml_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import os
from datetime import datetime

# Import ML pipeline
try:
    from glucose_pipeline import GlucosePredictor
except ImportError:
    import sys
    sys.path.append('/app')
    from glucose_pipeline import GlucosePredictor

app = FastAPI(
    title="Glucose Prediction API",
    description="ML API for predicting glucose levels",
    version="1.0.0"
)

# -------------------------------------------------
# CORS CONFIG (Fix for frontend CORS blocking)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allow everything for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Initialize ML predictor
# -------------------------------------------------
predictor = GlucosePredictor()

# -------------------------------------------------
# MODELS
# -------------------------------------------------
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

# -------------------------------------------------
# MIDDLEWARE DEBUG LOGGER
# (This was broken due to missing import -- NOW FIXED)
# -------------------------------------------------
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    print(f"🔍 Incoming: {request.method} {request.url}")

    if request.method == "POST" and "/predict" in str(request.url):
        body = await request.body()
        print(f"📝 Body: {body.decode()}")

    response = await call_next(request)
    print(f"📡 Status: {response.status_code}")
    return response

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
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
    try:
        print(f"🎯 Prediction request for patient: {request.patient_id}")

        input_data = {
            'time': request.time or datetime.now().isoformat(),
            'dosage': request.dosage,
            'cbg': request.cbg,
            'cbg_pre_meal': request.cbg,
            'carbohydrates_estimation': request.carbohydrates_estimation,
            'rice_cups': request.rice_cups,
            'meal_type': request.meal_type
        }

        print(f"📦 Input: {input_data}")

        result = predictor.predict(request.patient_id, input_data)
        print(f"✅ Prediction successful: {result}")

        return result

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/{patient_id}", response_model=TrainingResponse)
async def train_model(patient_id: str):
    try:
        return predictor.train_patient_model(patient_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{patient_id}")
async def get_model_status(patient_id: str):
    try:
        return predictor.get_status(patient_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/routes")
async def get_routes():
    return [
        {
            "path": route.path,
            "name": route.name,
            "methods": list(getattr(route, "methods", []))
        }
        for route in app.routes
    ]

# Debug POST test
@app.post("/simple_post_test")
async def simple_post(data: dict):
    return {"message": "POST test works", "data": data}

# -------------------------------------------------
# RAILWAY ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ml_api:app", host="0.0.0.0", port=port)
