import pandas as pd
import numpy as np
import os

def save_background_data():
    # Load processed test data
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    
    # Select a random sample of 100 instances for background
    np.random.seed(42)
    indices = np.random.choice(X_test.shape[0], 100, replace=False)
    background_data = X_test[indices]
    
    os.makedirs('firm_bankruptcy_prediction/data/processed', exist_ok=True)
    np.save('firm_bankruptcy_prediction/data/processed/shap_background.npy', background_data)
    print("SHAP background data saved.")

if __name__ == "__main__":
    save_background_data()
