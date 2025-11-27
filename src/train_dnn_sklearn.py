import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_dnn_sklearn():
    # Load data
    X_train = np.load('firm_bankruptcy_prediction/data/processed/X_train.npy')
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_train = np.load('firm_bankruptcy_prediction/data/processed/y_train.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    # Initialize MLPClassifier
    # Using a similar architecture: 64 -> 32
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2
    )
    
    # Train
    print("Training MLPClassifier...")
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
    joblib.dump(model, 'firm_bankruptcy_prediction/models/dnn_model_sklearn.joblib')
    print("Model saved to firm_bankruptcy_prediction/models/dnn_model_sklearn.joblib")

if __name__ == "__main__":
    train_dnn_sklearn()
