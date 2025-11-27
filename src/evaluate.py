import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

def evaluate_models():
    # Load data
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    print("Loading models...")
    
    # Load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('firm_bankruptcy_prediction/models/xgboost_model.json')
    
    # Load MLP (Sklearn)
    mlp_model = joblib.load('firm_bankruptcy_prediction/models/dnn_model_sklearn.joblib')
    
    # Predict
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_mlp = mlp_model.predict(X_test)
    
    # Metrics
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, average='macro')
    
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp, average='macro')
    
    print("\n" + "="*50)
    print("FINAL MODEL COMPARISON")
    print("="*50)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'F1-Score (Macro)':<15}")
    print("-" * 50)
    print(f"{'XGBoost':<20} | {acc_xgb:.4f}     | {f1_xgb:.4f}")
    print(f"{'DNN (MLP)':<20} | {acc_mlp:.4f}     | {f1_mlp:.4f}")
    print("-" * 50)
    
    print("\nTarget Metrics:")
    print("DNN: ~97.60%")
    print("XGBoost: ~96.89%")

if __name__ == "__main__":
    evaluate_models()
