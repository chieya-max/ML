import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import numpy as np
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
import logging

# ---------------- CONFIGURATION ----------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
PATIENT_ID_COL = "patient_id"
TIME_COL = "time"
TARGET = "cbg_post_meal"
MIN_RECORDS_PER_USER = 5  # Reduced for production flexibility
MODELS_DIR = os.getenv('MODELS_DIR', 'patient_models')
os.makedirs(MODELS_DIR, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- DATA FETCHING FUNCTIONS ----------------
def fetch_patient_demographics(patient_id: str) -> Optional[Dict]:
    """Fetch static patient information"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = supabase.table("patients").select("*").eq("id", patient_id).execute()
        return resp.data[0] if resp.data else None
    except Exception as e:
        logger.error(f"Error fetching demographics for patient {patient_id}: {str(e)}")
        return None

def fetch_patient_meals(patient_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch meal data for a patient"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (supabase.table("meals")
                .select("*")
                .eq("patient_id", patient_id)
                .order("time", desc=False)  # Ascending for temporal analysis
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching meals for patient {patient_id}: {str(e)}")
        return pd.DataFrame()

def fetch_patient_insulin(patient_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch insulin data for a patient"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (supabase.table("insulin")
                .select("*")
                .eq("patient_id", patient_id)
                .order("time", desc=False)
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching insulin for patient {patient_id}: {str(e)}")
        return pd.DataFrame()

def fetch_patient_activities(patient_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch activity data for a patient"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (supabase.table("activities")
                .select("*")
                .eq("patient_id", patient_id)
                .order("start_time", desc=False)
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching activities for patient {patient_id}: {str(e)}")
        return pd.DataFrame()

def fetch_patient_sleep(patient_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch sleep data for a patient"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (supabase.table("patient_sleep")
                .select("*")
                .eq("patient_id", patient_id)
                .order("recorded_at", desc=False)
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching sleep for patient {patient_id}: {str(e)}")
        return pd.DataFrame()

def fetch_patient_stress(patient_id: str, limit: int = 10000) -> pd.DataFrame:
    """Fetch stress data for a patient"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = (supabase.table("patient_stress")
                .select("*")
                .eq("patient_id", patient_id)
                .order("recorded_at", desc=False)
                .limit(limit)
                .execute())
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching stress for patient {patient_id}: {str(e)}")
        return pd.DataFrame()

def fetch_all_patient_ids() -> List[str]:
    """Get list of all patient IDs"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = supabase.table("patients").select("id").execute()
        return [patient["id"] for patient in resp.data] if resp.data else []
    except Exception as e:
        logger.error(f"Error fetching patient IDs: {str(e)}")
        return []

# ---------------- DATA INTEGRATION ----------------
def integrate_patient_data(patient_id: str) -> pd.DataFrame:
    """Integrate data for a single patient"""
    logger.info(f"Integrating data for patient: {patient_id}")
    
    try:
        # Fetch all data sources
        meals_df = fetch_patient_meals(patient_id)
        insulin_df = fetch_patient_insulin(patient_id)
        activities_df = fetch_patient_activities(patient_id)
        sleep_df = fetch_patient_sleep(patient_id)
        stress_df = fetch_patient_stress(patient_id)
        demographics = fetch_patient_demographics(patient_id)
        
        logger.info(f"Data counts - Meals: {len(meals_df)}, Insulin: {len(insulin_df)}, "
                   f"Activities: {len(activities_df)}, Sleep: {len(sleep_df)}, Stress: {len(stress_df)}")
        
        # Use insulin data as base (contains target variable)
        if insulin_df.empty:
            logger.warning(f"No insulin data for patient {patient_id}")
            return pd.DataFrame()
        
        base_df = insulin_df.copy()
        
        # Process datetime
        base_df[TIME_COL] = pd.to_datetime(base_df[TIME_COL], errors='coerce', utc=True)
        base_df = base_df[base_df[TIME_COL].notna()]
        
        if base_df.empty:
            logger.warning(f"No valid datetime data for patient {patient_id}")
            return pd.DataFrame()
            
        base_df = base_df.sort_values(TIME_COL).reset_index(drop=True)
        base_df[PATIENT_ID_COL] = patient_id
        
        # Check target data availability
        target_records = base_df[~base_df[TARGET].isna()]
        logger.info(f"Records with {TARGET}: {len(target_records)}")
        
        # Merge meals data
        if not meals_df.empty:
            meals_df[TIME_COL] = pd.to_datetime(meals_df[TIME_COL], errors='coerce', utc=True)
            meals_df = meals_df[meals_df[TIME_COL].notna()].sort_values(TIME_COL)
            
            for idx, insulin_row in base_df.iterrows():
                insulin_time = insulin_row[TIME_COL]
                time_diffs = (meals_df[TIME_COL] - insulin_time).abs()
                
                if not time_diffs.empty:
                    min_diff_idx = time_diffs.idxmin()
                    if time_diffs[min_diff_idx] <= timedelta(hours=2):
                        meal_row = meals_df.loc[min_diff_idx]
                        base_df.at[idx, 'carbs'] = meal_row.get('carbohydrates_estimation', 0)
                        base_df.at[idx, 'meal_type'] = meal_row.get('meal_type', 'unknown')
                        base_df.at[idx, 'rice_cups'] = meal_row.get('rice_cups', 0)
        
        # Add demographic data
        if demographics:
            for key, value in demographics.items():
                if key not in ['id', 'created_at'] and isinstance(value, (str, int, float)):
                    base_df[f"demographic_{key}"] = value
        
        logger.info(f"Final integrated records: {len(base_df)}")
        return base_df
        
    except Exception as e:
        logger.error(f"Error integrating data for {patient_id}: {str(e)}")
        return pd.DataFrame()

# ---------------- FEATURE ENGINEERING ----------------
def create_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for patient data"""
    if df.empty:
        return df
    
    try:
        df = df.copy()
        
        # Process datetime
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce', utc=True)
        df = df[df[TIME_COL].notna()]
        
        if df.empty:
            return df
            
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        
        # Handle carbohydrates
        if 'carbohydrates_estimation' in df.columns:
            df['carbs'] = pd.to_numeric(df['carbohydrates_estimation'], errors='coerce').fillna(0)
        else:
            df['carbs'] = 0
        
        # Time-based features
        df["hour"] = df[TIME_COL].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
        df["day_of_week"] = df[TIME_COL].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["month"] = df[TIME_COL].dt.month
        
        # Insulin features
        if 'dosage' in df.columns:
            df['insulin_dosage'] = df['dosage'].fillna(0)
            df['has_insulin'] = (df['insulin_dosage'] > 0).astype(int)
        
        # Meal type encoding
        if 'meal_type' in df.columns:
            df['meal_type_enc'] = pd.factorize(df['meal_type'].fillna('unknown'))[0]
        
        # Simple lag features
        numeric_cols = ['carbs', 'insulin_dosage'] if 'insulin_dosage' in df.columns else ['carbs']
        for col in numeric_cols:
            if col in df.columns:
                df[f"{col}_lag1"] = df[col].shift(1).fillna(0)
        
        # Fill remaining numeric NaNs
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != TARGET and col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        return pd.DataFrame()

# ---------------- MODEL TRAINING ----------------
def train_personalized_model(patient_id: str, patient_data: pd.DataFrame) -> Optional[Dict]:
    """Train a personalized model for a patient"""
    if patient_data.empty or TARGET not in patient_data.columns:
        logger.warning(f"No target data for patient {patient_id}")
        return None
    
    # Filter to valid target data
    valid_data = patient_data[~patient_data[TARGET].isna()]
    
    if len(valid_data) < MIN_RECORDS_PER_USER:
        logger.warning(f"Insufficient data for patient {patient_id}: {len(valid_data)} < {MIN_RECORDS_PER_USER}")
        return None
    
    # Select features
    exclude_cols = [PATIENT_ID_COL, TIME_COL, TARGET, 'id']
    feature_cols = [col for col in valid_data.columns 
                   if col not in exclude_cols and np.issubdtype(valid_data[col].dtype, np.number)]
    
    if len(feature_cols) < 2:
        logger.warning(f"Not enough features for patient {patient_id}: {len(feature_cols)}")
        return None
    
    X = valid_data[feature_cols]
    y = valid_data[TARGET]
    
    # Training strategy based on data size
    if len(valid_data) <= 15:
        # Small dataset strategy
        model = lgb.LGBMRegressor(
            n_estimators=20,
            learning_rate=0.15,
            num_leaves=5,
            max_depth=3,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1
        )
        
        if len(valid_data) <= 10:
            # Very small dataset - use all data
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            # Small dataset - temporal split
            split_idx = max(3, int(0.7 * len(valid_data)))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        # Standard dataset
        split_idx = int(0.7 * len(valid_data))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            random_state=42,
            verbosity=-1
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    
    logger.info(f"Model trained for {patient_id} - Train RMSE: {train_rmse:.1f}, Test RMSE: {test_rmse:.1f}")
    
    # Prepare model package
    model_info = {
        "model": model,
        "feature_cols": feature_cols,
        "training_records": len(valid_data),
        "training_data_stats": {
            'mean': y.mean(),
            'std': y.std(),
            'min': y.min(),
            'max': y.max()
        },
        "metrics": {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "test_mae": test_mae
        },
        "last_trained": datetime.now().isoformat(),
        "patient_id": patient_id,
        "model_version": "1.0"
    }
    
    return model_info

def save_patient_model(patient_id: str, model_info: Dict) -> str:
    """Save trained model to disk"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"model_{patient_id}.joblib")
    joblib.dump(model_info, model_path)
    logger.info(f"Model saved for patient {patient_id} at {model_path}")
    return model_path

def load_patient_model(patient_id: str) -> Optional[Dict]:
    """Load trained model from disk"""
    model_path = os.path.join(MODELS_DIR, f"model_{patient_id}.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error loading model for patient {patient_id}: {str(e)}")
    return None

# ---------------- PREDICTION SERVICE ----------------
def predict_for_patient(patient_id: str, input_data: Dict) -> Dict:
    """Predict glucose for a specific patient"""
    model_info = load_patient_model(patient_id)
    
    if not model_info:
        raise ValueError(f"No trained model found for patient {patient_id}")
    
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    
    # Create input DataFrame
    input_df = pd.DataFrame([input_data])
    input_df[PATIENT_ID_COL] = patient_id
    
    # Create features
    processed_data = create_patient_features(input_df)
    
    if processed_data.empty:
        raise ValueError("No features created from input data")
    
    # Handle missing features
    for feature in feature_cols:
        if feature not in processed_data.columns:
            processed_data[feature] = 0
    
    # Predict
    X = processed_data[feature_cols]
    prediction = model.predict(X)[0]
    
    return {
        "patient_id": patient_id,
        "predicted_glucose": float(prediction),
        "prediction_timestamp": datetime.now().isoformat(),
        "model_confidence": f"Trained on {model_info['training_records']} records",
        "expected_accuracy": f"±{model_info['metrics']['test_rmse']:.1f} mg/dL"
    }

# ---------------- MODEL MANAGEMENT ----------------
def train_all_personalized_models() -> Dict:
    """Train models for all patients with sufficient data"""
    logger.info("Starting model training pipeline")
    
    patient_ids = fetch_all_patient_ids()
    logger.info(f"Found {len(patient_ids)} patients")
    
    trained_models = {}
    
    for patient_id in patient_ids:
        try:
            logger.info(f"Processing patient: {patient_id}")
            
            patient_data = integrate_patient_data(patient_id)
            if patient_data.empty:
                continue
                
            featured_data = create_patient_features(patient_data)
            if featured_data.empty:
                continue
                
            model_info = train_personalized_model(patient_id, featured_data)
            if model_info:
                model_path = save_patient_model(patient_id, model_info)
                trained_models[patient_id] = {
                    "path": model_path,
                    "metrics": model_info["metrics"],
                    "records": model_info["training_records"],
                    "last_trained": model_info["last_trained"]
                }
                
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {str(e)}")
            continue
    
    logger.info(f"Training completed. Successfully trained {len(trained_models)} models")
    return trained_models

def get_model_status(patient_id: str) -> Dict:
    """Get status of a patient's model"""
    model_info = load_patient_model(patient_id)
    
    if model_info:
        return {
            "status": "trained",
            "patient_id": patient_id,
            "training_records": model_info["training_records"],
            "last_trained": model_info["last_trained"],
            "test_rmse": model_info["metrics"]["test_rmse"],
            "test_mae": model_info["metrics"]["test_mae"]
        }
    else:
        return {
            "status": "not_trained",
            "patient_id": patient_id,
            "message": "No trained model found"
        }

# ---------------- PRODUCTION API ----------------
class GlucosePredictor:
    """Production-ready glucose prediction service"""
    
    def __init__(self):
        self.models_dir = MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_patient_model(self, patient_id: str) -> Dict:
        """Train model for a specific patient"""
        try:
            patient_data = integrate_patient_data(patient_id)
            if patient_data.empty:
                return {"status": "error", "message": "No data available"}
                
            model_info = train_personalized_model(patient_id, patient_data)
            if model_info:
                save_patient_model(patient_id, model_info)
                return {
                    "status": "success",
                    "patient_id": patient_id,
                    "training_records": model_info["training_records"],
                    "metrics": model_info["metrics"]
                }
            else:
                return {"status": "error", "message": "Model training failed"}
                
        except Exception as e:
            logger.error(f"Error training model for {patient_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, patient_id: str, input_data: Dict) -> Dict:
        """Make prediction for a patient"""
        try:
            return predict_for_patient(patient_id, input_data)
        except Exception as e:
            logger.error(f"Error predicting for {patient_id}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_status(self, patient_id: str) -> Dict:
        """Get model status for a patient"""
        return get_model_status(patient_id)
    
    def retrain_all_models(self) -> Dict:
        """Retrain all patient models"""
        try:
            trained_models = train_all_personalized_models()
            return {
                "status": "success",
                "trained_models": len(trained_models),
                "details": trained_models
            }
        except Exception as e:
            logger.error(f"Error retraining all models: {str(e)}")
            return {"status": "error", "message": str(e)}

# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    """Command-line interface for the ML pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Glucose Prediction ML Pipeline')
    parser.add_argument('--train-all', action='store_true', help='Train models for all patients')
    parser.add_argument('--train-patient', type=str, help='Train model for specific patient ID')
    parser.add_argument('--predict', type=str, help='Patient ID for prediction')
    parser.add_argument('--status', type=str, help='Get model status for patient ID')
    
    args = parser.parse_args()
    
    predictor = GlucosePredictor()
    
    if args.train_all:
        print("🚀 Training models for all patients...")
        result = predictor.retrain_all_models()
        print(f"✅ {result}")
        
    elif args.train_patient:
        print(f"🚀 Training model for patient {args.train_patient}...")
        result = predictor.train_patient_model(args.train_patient)
        print(f"✅ {result}")
        
    elif args.predict:
        print(f"🎯 Getting prediction for patient {args.predict}...")
        # Example input data
        sample_input = {
            'time': datetime.now().isoformat(),
            'dosage': 5,
            'cbg': 120,
            'cbg_pre_meal': 120,
            'carbohydrates_estimation': 50,
            'rice_cups': 1,
            'meal_type': 'lunch'
        }
        result = predictor.predict(args.predict, sample_input)
        print(f"✅ {result}")
        
    elif args.status:
        print(f"📊 Getting status for patient {args.status}...")
        result = predictor.get_status(args.status)
        print(f"✅ {result}")
        
    else:
        print("🤖 Glucose Prediction ML Pipeline")
        print("Usage:")
        print("  --train-all          Train models for all patients")
        print("  --train-patient ID   Train model for specific patient")
        print("  --predict ID         Get prediction for patient")
        print("  --status ID          Get model status for patient")