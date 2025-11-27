import xgboost as xgb
import pandas as pd
import numpy as np

def get_top_features():
    # Load model
    model = xgb.XGBClassifier()
    model.load_model('firm_bankruptcy_prediction/models/xgboost_model.json')
    
    # Load feature names (we need to get them from the original dataframe)
    df = pd.read_csv('firm_bankruptcy_prediction/data/data.csv')
    feature_names = df.drop('Bankrupt?', axis=1).columns.tolist()
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 Features:")
    for i in range(10):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    get_top_features()
