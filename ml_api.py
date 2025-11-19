# ml_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import os
from datetime import datetime


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

@app.get("/health")
def health():
    return {"status": "ok"}
    
# CORS middleware - allow your React app domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
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
        print(f"🎯 Received prediction request for patient: {request.patient_id}")
        
        input_data = {
            'time': request.time or datetime.now().isoformat(),
            'dosage': request.dosage,
            'cbg': request.cbg,
            'cbg_pre_meal': request.cbg,
            'carbohydrates_estimation': request.carbohydrates_estimation,
            'rice_cups': request.rice_cups,
            'meal_type': request.meal_type
        }
        
        print(f"📦 Input data: {input_data}")
        
        result = predictor.predict(request.patient_id, input_data)
        print(f"✅ Prediction successful: {result}")
        
        return result
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
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

@app.get("/routes")
async def get_routes():
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": list(route.methods) if hasattr(route, 'methods') else []
        })
    return routes

@app.post("/test_predict")
async def test_predict(request: PredictionRequest):
    """Test endpoint to debug prediction issues"""
    try:
        print("🔍 Received prediction request:", request.dict())
        
        input_data = {
            'time': request.time or datetime.now().isoformat(),
            'dosage': request.dosage,
            'cbg': request.cbg,
            'cbg_pre_meal': request.cbg,
            'carbohydrates_estimation': request.carbohydrates_estimation,
            'rice_cups': request.rice_cups,
            'meal_type': request.meal_type
        }
        
        print("📦 Processed input data:", input_data)
        
        # Test the predictor
        result = predictor.predict(request.patient_id, input_data)
        print("✅ Prediction result:", result)
        
        return {"status": "success", "test_prediction": result}
    except Exception as e:
        print("❌ Prediction error:", str(e))
        raise HTTPException(status_code=400, detail=f"Test prediction failed: {str(e)}")

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    print(f"🔍 Incoming request: {request.method} {request.url}")
    print(f"📦 Headers: {dict(request.headers)}")
    
    if request.method == "POST" and "/predict" in str(request.url):
        body = await request.body()
        print(f"📝 Request body: {body.decode()}")
    
    response = await call_next(request)
    print(f"📡 Response status: {response.status_code}")
    return response

@app.post("/debug_predict")
async def debug_predict(request: PredictionRequest):
    """Debug endpoint to see if the issue is with the model"""
    try:
        print("🎯 Debug endpoint called successfully")
        print(f"📦 Received data: {request.dict()}")
        
        # Return a simple success response to test if the endpoint works
        return {
            "patient_id": request.patient_id,
            "predicted_glucose": 150.0,  # Mock data
            "prediction_timestamp": datetime.now().isoformat(),
            "model_confidence": "High",
            "expected_accuracy": "±10 mg/dL",
            "debug": True
        }
    except Exception as e:
        print(f"❌ Debug endpoint error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# For Railway deployment

if __name__ == "__main__":
    # Read Railway's PORT env variable, default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("ml_api:app", host="0.0.0.0", port=port)
