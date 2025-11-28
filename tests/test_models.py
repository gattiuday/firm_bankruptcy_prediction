import pytest
import numpy as np
import xgboost as xgb
import joblib
import os

def test_models_exist():
    """Test that model files exist."""
    assert os.path.exists('models/xgboost_model_tuned.json')
    assert os.path.exists('models/dnn_model_sklearn_tuned.joblib')

def test_xgboost_prediction():
    """Test XGBoost prediction shape and range."""
    # Load model
    model = xgb.XGBClassifier()
    model.load_model('models/xgboost_model_tuned.json')
    
    # Create dummy input
    dummy_input = np.random.rand(1, 95)
    
    # Predict
    pred = model.predict(dummy_input)
    prob = model.predict_proba(dummy_input)
    
    assert pred.shape == (1,)
    assert prob.shape == (1, 2)
    assert 0 <= prob[0][1] <= 1

def test_mlp_prediction():
    """Test MLP prediction shape and range."""
    # Load model
    model = joblib.load('models/dnn_model_sklearn_tuned.joblib')
    
    # Create dummy input
    dummy_input = np.random.rand(1, 95)
    
    # Predict
    pred = model.predict(dummy_input)
    prob = model.predict_proba(dummy_input)
    
    assert pred.shape == (1,)
    assert prob.shape == (1, 2)
    assert 0 <= prob[0][1] <= 1
