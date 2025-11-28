import os

print("DEBUG: Imports started")
try:
    import streamlit as st
    print("DEBUG: Streamlit imported")
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    print(f"DEBUG: XGBoost version {xgb.__version__}")
    import joblib
    print("DEBUG: Imports completed")
except Exception as e:
    print(f"DEBUG: Error during imports: {e}")
    raise e

# Set page config
st.set_page_config(page_title="Firm Bankruptcy Prediction", layout="wide")

# Load assets
@st.cache_resource
def load_assets():
    # Load Scaler
    scaler = joblib.load('data/processed/scaler.joblib')
    
    # Load Means
    means = joblib.load('data/processed/feature_means.joblib')
    
    # Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('models/xgboost_model_tuned.json')
    
    # Load MLP
    mlp_model = joblib.load('models/dnn_model_sklearn_tuned.joblib')
    
    return scaler, means, xgb_model, mlp_model

scaler, means, xgb_model, mlp_model = load_assets()

# Title
st.title("Firm Bankruptcy Prediction System")
st.markdown("""
This system predicts the probability of firm bankruptcy using **XGBoost** and **Deep Neural Networks (MLP)**.
Adjust the key financial metrics in the sidebar to see how they impact the risk.
""")

# Sidebar Inputs (Top 10 Features)
st.sidebar.header("Key Financial Metrics")

def user_input_features():
    # Top 10 features identified by importance
    # We use the mean as default value
    
    f1 = st.sidebar.slider("Continuous interest rate (after tax)", 0.0, 1.0, float(means[' Continuous interest rate (after tax)']))
    f2 = st.sidebar.slider("Borrowing dependency", 0.0, 1.0, float(means[' Borrowing dependency']))
    f3 = st.sidebar.slider("Debt ratio %", 0.0, 1.0, float(means[' Debt ratio %']))
    f4 = st.sidebar.slider("Persistent EPS in the Last Four Seasons", 0.0, 1.0, float(means[' Persistent EPS in the Last Four Seasons']))
    f5 = st.sidebar.slider("Interest Expense Ratio", 0.0, 1.0, float(means[' Interest Expense Ratio']))
    f6 = st.sidebar.slider("Total debt/Total net worth", 0.0, 1.0, float(means[' Total debt/Total net worth']))
    f7 = st.sidebar.slider("Net worth/Assets", 0.0, 1.0, float(means[' Net worth/Assets']))
    f8 = st.sidebar.slider("Net Income to Total Assets", 0.0, 1.0, float(means[' Net Income to Total Assets']))
    f9 = st.sidebar.slider("Working Capital to Total Assets", 0.0, 1.0, float(means[' Working Capital to Total Assets']))
    f10 = st.sidebar.slider("Operating Profit Rate", 0.0, 1.0, float(means[' Operating Profit Rate']))
    
    return {
        ' Continuous interest rate (after tax)': f1,
        ' Borrowing dependency': f2,
        ' Debt ratio %': f3,
        ' Persistent EPS in the Last Four Seasons': f4,
        ' Interest Expense Ratio': f5,
        ' Total debt/Total net worth': f6,
        ' Net worth/Assets': f7,
        ' Net Income to Total Assets': f8,
        ' Working Capital to Total Assets': f9,
        ' Operating Profit Rate': f10
    }

input_dict = user_input_features()

# Prepare input vector
# Start with means for all features
input_data = means.copy()
# Update with user inputs
for key, value in input_dict.items():
    input_data[key] = value

# Reshape and Scale
input_vector = input_data.values.reshape(1, -1)
input_scaled = scaler.transform(input_vector)

# Prediction
col1, col2 = st.columns(2)

with col1:
    st.subheader("XGBoost Prediction")
    prob_xgb = xgb_model.predict_proba(input_scaled)[0][1]
    st.metric("Bankruptcy Probability", f"{prob_xgb:.2%}")
    if prob_xgb > 0.5:
        st.error("High Risk of Bankruptcy")
    else:
        st.success("Low Risk")

with col2:
    st.subheader("DNN (MLP) Prediction")
    prob_mlp = mlp_model.predict_proba(input_scaled)[0][1]
    st.metric("Bankruptcy Probability", f"{prob_mlp:.2%}")
    if prob_mlp > 0.5:
        st.error("High Risk of Bankruptcy")
    else:
        st.success("Low Risk")

# Feature Importance Plot (Static for context)
st.markdown("---")
st.subheader("Model Insights")
st.info("The inputs above correspond to the top 10 most important features driving the XGBoost model's decisions.")

# --- SHAP Explanation ---
import shap
import matplotlib.pyplot as plt

st.markdown("### Explainability (SHAP)")
st.write("Understanding *why* the model made this prediction.")

if st.button("Explain Prediction"):
    with st.spinner("Calculating SHAP values..."):
        # Load background data for SHAP (optional for TreeExplainer but good practice)
        # For TreeExplainer we can just use the model
        explainer = shap.TreeExplainer(xgb_model)
        
        # Calculate SHAP values for the input
        # input_scaled is (1, 95)
        shap_values = explainer(input_scaled)
        
        # We need feature names for the plot
        # The scaler returns numpy array, so we lose column names.
        # We need to map them back. 
        # Since we used all features in training, we need the full list of 95 features.
        # However, we only have the top 10 in the UI. 
        # The input_scaled has all 95 features (filled with means).
        
        # Let's get feature names from the original dataset columns
        # We can load them from the means index
        feature_names = means.index.tolist()
        
        # Waterfall plot
        st.subheader("Waterfall Plot")
        st.write("This plot shows how each feature contributed to pushing the prediction from the base value to the final output.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        
        # Force Plot (Matplotlib version if available, or just stick to waterfall which is clearer)
        # Waterfall is usually better for single prediction explanation.
