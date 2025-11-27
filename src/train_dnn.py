import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def train_dnn():
    # Load data
    X_train = np.load('firm_bankruptcy_prediction/data/processed/X_train.npy')
    X_test = np.load('firm_bankruptcy_prediction/data/processed/X_test.npy')
    y_train = np.load('firm_bankruptcy_prediction/data/processed/y_train.npy')
    y_test = np.load('firm_bankruptcy_prediction/data/processed/y_test.npy')
    
    # Calculate class weights
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class Weights: {class_weight}")
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    os.makedirs('firm_bankruptcy_prediction/models', exist_ok=True)
    model.save('firm_bankruptcy_prediction/models/dnn_model.keras')
    print("Model saved to firm_bankruptcy_prediction/models/dnn_model.keras")

if __name__ == "__main__":
    train_dnn()
