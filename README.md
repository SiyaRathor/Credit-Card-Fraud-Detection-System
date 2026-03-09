# Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent credit card transactions using classification models and deploys the best performing model using Streamlit.

## Project Overview

This project builds a fraud detection system using a real-world credit card transaction dataset. The model learns patterns of fraudulent transactions and predicts whether a transaction is fraud or legitimate.

The workflow includes:

- Data preprocessing
- Feature scaling
- Handling class imbalance using SMOTE
- Training multiple classification models
- Model evaluation using ROC-AUC and Precision-Recall metrics
- Model comparison
- Feature importance analysis
- Deployment using Streamlit


## Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

- Source: Kaggle
- Transactions: 284,807
- Fraud cases: 492

The dataset is highly imbalanced which makes fraud detection a challenging problem.

---

## Models Used

| Model | Description |
|------|-------------|
| Logistic Regression | Baseline classification model |
| Random Forest | Ensemble model for better fraud detection |

Random Forest performed better and was used as the final deployed model.


## Model Evaluation Metrics

Due to class imbalance, accuracy alone is not reliable. Therefore the following metrics were used:

- ROC-AUC Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Model Evaluation Visualizations

### ROC Curve
Shows model performance across different thresholds.

### Precision-Recall Curve
Important for evaluating models on imbalanced datasets.

### Feature Importance
Identifies which transaction features contribute most to fraud detection.


## Deployment

The trained model is deployed using **Streamlit** as an interactive web application.

Users can input transaction features and the model predicts whether the transaction is **fraudulent or legitimate**.

---

## Run Locally

Install dependencies:
pip install -r requirements.txt


Run the application:
streamlit run app.py


## Project Structure

Credit-Card-Fraud-Detection-System
│
├── app.py
├── frauddetection.py
├── fraud_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── creditcard.csv
├── requirements.txt
├── README.md
└── screenshots/


---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit
- Joblib

## Key Learnings

- Handling imbalanced datasets using SMOTE
- Importance of ROC-AUC and Precision-Recall metrics
- Model comparison for classification problems
- Feature importance interpretation
- Deploying ML models using Streamlit

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Trying advanced models like XGBoost or LightGBM
- Real-time fraud detection pipeline
- Cloud deployment

