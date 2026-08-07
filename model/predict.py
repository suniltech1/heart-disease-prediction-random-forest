import os
import joblib
import pandas as pd
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'saved_model', 'random_forest.pkl')

def load_model() -> Dict[str, Any]:
    """
    Loads the trained model and associated metadata (like feature names).
    Returns a dictionary containing the pipeline and metadata.
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        logging.error(f"Model file not found at {MODEL_SAVE_PATH}")
        raise FileNotFoundError(f"Trained model not found at {MODEL_SAVE_PATH}. Please run train_model.py first.")
    
    try:
        model_data = joblib.load(MODEL_SAVE_PATH)
        logging.info("Model loaded successfully.")
        return model_data
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        raise e

def _determine_risk_level(probability: float) -> str:
    """
    Determines the risk level based on the probability.
    High >= 0.80, Moderate >= 0.60, Low otherwise.
    """
    if probability >= 0.80:
        return "High"
    elif probability >= 0.60:
        return "Moderate"
    else:
        return "Low"

def predict_patient(patient_data: dict, model_data: dict) -> Tuple[str, float, str, str]:
    """
    Predicts heart disease for a given patient.
    
    Args:
        patient_data (dict): A dictionary where keys are feature names and values are patient values.
        model_data (dict): The loaded model dictionary containing the pipeline.
        
    Returns:
        Tuple containing:
        - Prediction (str): 'Heart Disease' or 'No Heart Disease'
        - Probability (float): The probability of having heart disease (between 0 and 1)
        - Confidence (str): formatted as percentage 'XX%'
        - Risk Level (str): 'High', 'Moderate', or 'Low'
    """
    pipeline = model_data['pipeline']
    expected_features = model_data.get('feature_names_in', [])

    # Convert user-friendly sex values back to numeric format before prediction.
    if 'sex' in patient_data and isinstance(patient_data['sex'], str):
        sex_lower = patient_data['sex'].strip().lower()
        if sex_lower == 'male':
            patient_data['sex'] = 1
        elif sex_lower == 'female':
            patient_data['sex'] = 0

    # Ensure patient data aligns with expected features
    # If the user hasn't provided some features, we can try to fill with NaN or raise error.
    # We will construct a DataFrame with the exact columns.
    df = pd.DataFrame([patient_data])
    
    # Missing columns will be handled by the pipeline's imputer if they are completely missing,
    # but we should at least ensure the dataframe has the correct shape.
    for col in expected_features:
        if col not in df.columns:
            df[col] = None
            
    # Reorder to match training
    df = df[expected_features]
    
    # Predict
    pred_class = pipeline.predict(df)[0]
    
    # Predict probability (probability of class 1)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(df)[0]
        # Assuming class 1 is Heart Disease
        prob = probabilities[1]
    else:
        # Fallback if probability isn't supported (not applicable for RF, but good practice)
        prob = 1.0 if pred_class == 1 else 0.0

    # Format the outputs
    prediction = "Heart Disease" if pred_class == 1 else "No Heart Disease"
    
    # Confidence is typically the probability of the *predicted* class
    # E.g. if prob = 0.1, it's 90% confident in No Heart Disease.
    # But often in clinical settings, Confidence means probability of the positive class.
    # Based on the user prompt: "Probability >= 0.80 High", we will use `prob` directly for Risk.
    # Let's map confidence to the probability of the predicted outcome.
    confidence_val = prob if pred_class == 1 else (1 - prob)
    confidence = f"{confidence_val * 100:.0f}%"
    
    risk_level = _determine_risk_level(prob)
    
    return prediction, prob, confidence, risk_level

if __name__ == "__main__":
    # Simple test for predict.py
    try:
        model = load_model()
        # Mock patient based on dummy data generated in train_model.py
        mock_patient = {
            'age': 55,
            'sex': 'M',
            'cholesterol': 240,
            'blood_pressure': 135
        }
        
        prediction, prob, conf, risk = predict_patient(mock_patient, model)
        print("\n--- Prediction Results ---")
        print(f"Prediction: {prediction}")
        print(f"Model Probability: {prob:.2f}")
        print(f"Confidence: {conf}")
        print(f"Risk Level: {risk}")
        
    except Exception as e:
        print(f"Failed to test prediction: {e}")
