import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_xgboost():
    # Load data
    X_train = np.load('firm_bankruptcy_prediction/data/processed/X_train.npy')
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_train = np.load('firm_bankruptcy_prediction/data/processed/y_train.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    # Calculate scale_pos_weight
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale_pos_weight = neg / pos
    print(f"Scale Pos Weight: {scale_pos_weight}")
    
    # Initialize XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    os.makedirs('firm_bankruptcy_prediction/models', exist_ok=True)
    model.save_model('firm_bankruptcy_prediction/models/xgboost_model.json')
    print("Model saved to firm_bankruptcy_prediction/models/xgboost_model.json")

if __name__ == "__main__":
    train_xgboost()
