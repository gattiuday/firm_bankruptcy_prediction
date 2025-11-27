import pytest
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def test_data_files_exist():
    """Test that processed data files exist."""
    assert os.path.exists('firm_bankruptcy_prediction/data/processed/X_train.npy')
    assert os.path.exists('firm_bankruptcy_prediction/data/processed/X_test.npy')
    assert os.path.exists('firm_bankruptcy_prediction/data/processed/y_train.npy')
    assert os.path.exists('firm_bankruptcy_prediction/data/processed/y_test.npy')

def test_data_shapes():
    """Test that data shapes are consistent."""
    X_train = np.load('firm_bankruptcy_prediction/data/processed/X_train.npy')
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_train = np.load('firm_bankruptcy_prediction/data/processed/y_train.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    # Check feature count (should be 95 after dropping target)
    assert X_train.shape[1] == 95

def test_scaler_exists():
    """Test that the scaler is saved."""
    assert os.path.exists('firm_bankruptcy_prediction/data/processed/scaler.joblib')
