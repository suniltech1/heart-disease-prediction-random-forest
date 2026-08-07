import os
import sys
import pandas as pd
import streamlit as st
from PIL import Image

# Add the parent directory to sys.path so we can import model modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model.predict import load_model, predict_patient

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Prediction AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'heart.csv')
DATASET_ALT_PATH = os.path.join(BASE_DIR, 'dataset', 'heart_cleveland_upload-selected-columns.csv')
if not os.path.exists(DATASET_PATH) and os.path.exists(DATASET_ALT_PATH):
    DATASET_PATH = DATASET_ALT_PATH
FEATURE_IMPORTANCE_IMG = os.path.join(BASE_DIR, 'images', 'feature_importance_random_forest.png')

@st.cache_data
def load_dataset_schema():
    """
    Attempts to load the dataset to determine categories for dynamic form generation.
    Returns a dataframe if successful, else None.
    """
    if os.path.exists(DATASET_PATH):
        try:
            return pd.read_csv(DATASET_PATH)
        except Exception:
            return None
    return None

def main():
    # --- Sidebar ---
    st.sidebar.title("❤️ AI Heart Disease Prediction")
    st.sidebar.markdown("""
  
    
    This application uses an Artificial Intelligence model (Random Forest Classifier) to predict the likelihood of heart disease based on medical attributes.
    
    ### How to use:
    1. Enter the patient's medical details in the form.
    2. Click the **Predict** button.
    3. Review the diagnosis, confidence, and risk level.
    
    *Disclaimer: This is an academic project and not a certified medical tool. Always consult a healthcare professional for medical advice.*
    """)

    # --- Main Page ---
    st.title("Patient Information Form")
    st.markdown("Please fill out the patient's details below to generate a prediction.")

    # Load Model
    try:
        model_data = load_model()
        st.success("✅ Machine Learning Model Loaded Successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please ensure you have run `python model/train_model.py` first to generate the model.")
        return

    expected_features = model_data.get('feature_names_in', [])
    num_cols = model_data.get('numerical_columns', [])
    cat_cols = model_data.get('categorical_columns', [])
    
    df_schema = load_dataset_schema()
    
    if not expected_features:
        st.warning("Model metadata is missing feature names. Cannot generate form.")
        return

    # --- Dynamic Form Generation ---
    st.markdown("### 📋 Enter Patient Details")
    
    # We will use columns to make the form look modern
    col1, col2 = st.columns(2)
    
    patient_data = {}
    
    with st.form("patient_form"):
        for i, feature in enumerate(expected_features):
            # Alternate columns for inputs
            current_col = col1 if i % 2 == 0 else col2
            
            with current_col:
                # Format label to look nice
                label = feature.replace('_', ' ').title()
                
                if feature in num_cols:
                    # Number input
                    # Try to get min/max/mean from dataset if available for better default
                    default_val = 0.0
                    if df_schema is not None and feature in df_schema.columns:
                        default_val = float(df_schema[feature].mean())
                        
                    patient_data[feature] = st.number_input(
                        f"{label}", 
                        value=float(default_val), 
                        step=1.0
                    )
                    
                elif feature in cat_cols:
                    # Categorical input - Dropdown
                    if feature == 'sex':
                        options = ['Male', 'Female']
                    else:
                        options = [""]
                        if df_schema is not None and feature in df_schema.columns:
                            # Convert options to string to ensure json serialization isn't an issue
                            options = df_schema[feature].dropna().unique().tolist()
                    patient_data[feature] = st.selectbox(
                        f"{label}", 
                        options=options,
                        index=0 if options else None
                    )
                else:
                    # Fallback text input
                    patient_data[feature] = st.text_input(f"{label}")

        submit_button = st.form_submit_button(label="🔮 Predict Heart Disease")

    # --- Prediction Logic ---
    if submit_button:
        # Validation
        missing_fields = [k for k, v in patient_data.items() if v == "" or v is None]
        if missing_fields:
            st.error(f"⚠️ Please fill in all fields. Missing: {', '.join(missing_fields)}")
        else:
            with st.spinner("Analyzing patient data..."):
                try:
                    prediction, prob, confidence, risk_level = predict_patient(patient_data, model_data)
                    
                    st.markdown("---")
                    st.header("📊 Prediction Results")
                    
                    # Display metrics
                    m1, m2, m3, m4 = st.columns(4)
                    
                    if prediction == "Heart Disease":
                        m1.metric("Diagnosis", "🚨 Positive", delta="High Risk", delta_color="inverse")
                    else:
                        m1.metric("Diagnosis", "✅ Negative", delta="Low Risk", delta_color="normal")
                        
                    m2.metric("Confidence", confidence)
                    m3.metric("Risk Level", risk_level)
                    m4.metric("Model Probability", f"{prob:.2f}")

                    # Risk level styling
                    if risk_level == "High":
                        st.error("**Risk Level: HIGH** - Immediate medical consultation recommended.")
                    elif risk_level == "Moderate":
                        st.warning("**Risk Level: MODERATE** - Regular monitoring advised.")
                    else:
                        st.success("**Risk Level: LOW** - Keep up the healthy lifestyle!")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    # --- Feature Importance Visualization ---
    st.markdown("---")
    st.subheader("🔍 Model Interpretability")
    st.markdown("The chart below shows which features the Random Forest model considers most important when making a prediction.")
    
    if os.path.exists(FEATURE_IMPORTANCE_IMG):
        try:
            image = Image.open(FEATURE_IMPORTANCE_IMG)
            st.image(image, caption="Feature Importance (Random Forest)", use_container_width=True)
        except Exception as e:
            st.warning("Could not load feature importance chart.")
    else:
        st.info("Train the model to see the Feature Importance chart here.")

if __name__ == "__main__":
    main()
