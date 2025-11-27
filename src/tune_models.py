import numpy as np
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def tune_models():
    # Load data
    X_train = np.load('firm_bankruptcy_prediction/data/processed/X_train.npy')
    y_train = np.load('firm_bankruptcy_prediction/data/processed/y_train.npy')
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    # --- XGBoost Tuning ---
    print("Tuning XGBoost...")
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [1, 10, 30, 50],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
    xgb_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=20, scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    
    print(f"Best XGBoost Params: {xgb_search.best_params_}")
    print(f"Best XGBoost Score: {xgb_search.best_score_:.4f}")
    
    # Save best XGBoost
    best_xgb = xgb_search.best_estimator_
    best_xgb.save_model('firm_bankruptcy_prediction/models/xgboost_model_tuned.json')
    
    # --- MLP Tuning ---
    print("\nTuning MLP...")
    mlp_param_grid = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (100,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [200, 300]
    }
    
    mlp_model = MLPClassifier(random_state=42, early_stopping=True)
    mlp_search = RandomizedSearchCV(mlp_model, mlp_param_grid, n_iter=10, scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1)
    mlp_search.fit(X_train, y_train)
    
    print(f"Best MLP Params: {mlp_search.best_params_}")
    print(f"Best MLP Score: {mlp_search.best_score_:.4f}")
    
    # Save best MLP
    best_mlp = mlp_search.best_estimator_
    joblib.dump(best_mlp, 'firm_bankruptcy_prediction/models/dnn_model_sklearn_tuned.joblib')
    
    # --- Evaluation on Test Set ---
    print("\nFinal Evaluation on Test Set:")
    
    y_pred_xgb = best_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"Tuned XGBoost Accuracy: {acc_xgb:.4f}")
    
    y_pred_mlp = best_mlp.predict(X_test)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    print(f"Tuned MLP Accuracy: {acc_mlp:.4f}")

if __name__ == "__main__":
    tune_models()
