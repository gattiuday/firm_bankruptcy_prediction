# Firm Bankruptcy Prediction

This project implements predictive models to forecast firm bankruptcy using the Taiwanese Bankruptcy Prediction dataset. It achieves high accuracy using XGBoost and Deep Neural Networks (DNN).

## Project Overview

- **Dataset**: Taiwanese Bankruptcy Prediction (UCI Machine Learning Repository)
- **Models**: XGBoost, Deep Neural Network (TensorFlow/Keras)
- **Goal**: Predict whether a company will go bankrupt based on financial features.

## Results

| Model | Accuracy | F1-Score (Macro) |
|-------|----------|------------------|
| XGBoost (Tuned) | 97.07% | ~0.75 |
| DNN (MLP Tuned) | 96.48% | ~0.65 |

*Note: The original project achieved 97.60% (DNN) and 96.89% (XGBoost). Our tuned XGBoost model has surpassed the original XGBoost target.*

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Fetch Data**:
    ```bash
    python firm_bankruptcy_prediction/src/fetch_data.py
    ```

3.  **Preprocessing**:
    ```bash
    python firm_bankruptcy_prediction/src/preprocess.py
    ```

## Training

### XGBoost
```bash
python firm_bankruptcy_prediction/src/train_xgboost.py
```

### Deep Neural Network (MLP)
```bash
python firm_bankruptcy_prediction/src/train_dnn_sklearn.py
```

### Web Interface
To run the interactive demonstration app:
```bash
streamlit run firm_bankruptcy_prediction/app.py
```

### Docker Deployment
To build and run the application using Docker:
```bash
# Build the image
docker build -t firm-bankruptcy-prediction firm_bankruptcy_prediction

# Run the container
docker run -p 8501:8501 firm-bankruptcy-prediction
```

### Testing
To run the automated unit tests:
```bash
pytest firm_bankruptcy_prediction/tests
```

## Directory Structure
- `data/`: Dataset and processed files.
- `src/`: Source code for fetching, preprocessing, and training.
- `notebooks/`: Exploratory analysis.
- `models/`: Saved models.
- `app.py`: Streamlit web application.
- `Dockerfile`: Docker configuration.
- `tests/`: Unit tests.
