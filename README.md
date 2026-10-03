# Titanic Survival Classification

Course Project - Machine Learning & Data Science, FTN Novi Sad

Binary classification of Titanic passenger survival. The project includes data preprocessing, Exploratory Data Analysis (EDA), anomaly detection, training and comparison of 7 Machine Learning models (Random Forest, Decision Tree, Gradient Boosting, KNN, SVM, Logistic Regression, Naive Bayes), Grid Search hyperparameter optimization, feature importance analysis, and deployment via a Streamlit web application.

## Directory Structure

- `data/` - Dataset directory
- `src/`
  - `data_preparation.py` - Data loading and preprocessing module (contains prepare_data() function used by `train.py` and `feature_selection.py`)
  - `eda.py` - Exploratory Data Analysis
  - `anomaly_detection.py` - Anomaly detection (IQR, Isolation Forest)
  - `evaluate.py` - Model evaluation utilities (metrics, confusion matrix, feature importance), used by `train.py`
  - `train.py` - Grid Search, 5-fold cross-validation, model training, and final test set evaluation
  - `feature_selection.py` - Feature importance analysis and evaluation (all vs. top features)
  - `export_model.py` - Final model serialization
- `models/` - Exported (saved) trained models
- `results/` - Generated plots and metric logs
- `app/` - Streamlit application interface

> Note: `data_preparation.py` and `evaluate.py` are helper modules and should not be executed directly - their functions are imported by other scripts (`train.py`, `feature_selection.py`).

## Getting Started
```bash
uv sync
```
Data Analysis:
```bash
uv run python src/eda.py
uv run python src/anomaly_detection.py
```
Training, Grid Search, and Model Evaluation:
```bash
uv run python src/train.py
```
Feature Importance Analysis:
```bash
uv run python src/feature_selection.py
```
Deployment - Exporting Model & Running Streamlit App:
```bash
uv run python src/export_model.py
uv run streamlit run app/streamlit_app.py
```

## Methodology

Models were evaluated and compared using Grid Search with 5-fold cross-validation on the training dataset. The test set was kept isolated and evaluated only once on the selected best-performing model.

## Final Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** (max_depth=5, n_estimators=100) | **82.68%** | 81.16% | 75.68% | **78.32%** |

**Top Predictive Features:** Sex, Title, Fare, Passenger Class, and Age.
