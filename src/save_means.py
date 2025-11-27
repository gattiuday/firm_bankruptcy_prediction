import pandas as pd
import numpy as np
import joblib
import os

def save_means():
    df = pd.read_csv('firm_bankruptcy_prediction/data/data.csv')
    X = df.drop('Bankrupt?', axis=1)
    means = X.mean()
    
    os.makedirs('firm_bankruptcy_prediction/data/processed', exist_ok=True)
    joblib.dump(means, 'firm_bankruptcy_prediction/data/processed/feature_means.joblib')
    print("Feature means saved.")

if __name__ == "__main__":
    save_means()
