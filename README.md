# Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent credit card transactions using classification models and deploys the best performing model using **Streamlit**.

This system analyzes transaction patterns and predicts whether a transaction is **fraudulent or legitimate**.

---

# Live Demo

Deployed Web App:

https://your-app-name.streamlit.app

---

# Project Overview

Credit card fraud detection is a critical real-world machine learning problem because fraudulent transactions are extremely rare compared to normal transactions.

This project builds a complete **machine learning pipeline** that:

1. Processes real-world transaction data
2. Handles highly imbalanced classes
3. Trains multiple classification models
4. Evaluates models using advanced metrics
5. Deploys the final model as an interactive web application

---

# Machine Learning Workflow

The project follows the complete ML pipeline:

- Data preprocessing
- Feature scaling
- Handling class imbalance using **SMOTE**
- Training classification models
- Model evaluation using **ROC-AUC and Precision-Recall metrics**
- Model comparison
- Feature importance analysis
- Deployment using **Streamlit**

---

# Dataset

Dataset used: **Credit Card Fraud Detection Dataset**

Source: Kaggle

Key characteristics:

- Total transactions: **284,807**
- Fraud cases: **492**
- Fraud percentage: **0.17%**

This makes it a **highly imbalanced dataset**, which is one of the main challenges in fraud detection problems.

---

# Models Used

| Model | Description |
|------|-------------|
| Logistic Regression | Baseline classification model |
| Random Forest | Ensemble tree-based model for improved fraud detection |

After comparison, **Random Forest performed better** and was selected as the final model.

---

# Model Evaluation Metrics

Because the dataset is highly imbalanced, **accuracy alone is misleading**.

Therefore the following evaluation metrics were used:

- ROC-AUC Score
- Precision
- Recall
- F1 Score
- Confusion Matrix

These metrics help evaluate how well the model detects **rare fraud cases**.

---

# Model Evaluation Visualizations

## ROC Curve

Shows the model's ability to distinguish between fraud and legitimate transactions.

![ROC Curve](screenshots/roc_curve.png)

---

## Precision-Recall Curve

Important for evaluating performance on **imbalanced datasets**.

![Precision Recall Curve](screenshots/pr_curve.png)

---

## Feature Importance

Shows the most influential features used by the Random Forest model.

![Feature Importance](screenshots/feature_importance.png)

---

# Web Application

The trained model is deployed using **Streamlit**.

The application allows users to:

- Input transaction feature values
- Run the fraud detection model
- Receive an instant prediction

Prediction output:

- Legitimate Transaction
- Fraudulent Transaction

---

# Running the Project Locally

### Install Dependencies

```bash
pip install -r requirements.txt

# Run the Application
streamlit run app.py

# Project Structure
Credit-Card-Fraud-Detection-System
│
├── app.py                     # Streamlit web application
├── frauddetection.py          # Model training script
├── fraud_model.pkl            # Trained model
├── scaler.pkl                 # Feature scaler
├── model_columns.pkl          # Feature list
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── screenshots/               # Visualization images

# Technologies Used

Python

Pandas

NumPy

Scikit-learn

Imbalanced-learn (SMOTE)

Matplotlib

Streamlit

Joblib

# Key Learnings

Through this project I learned:

Handling highly imbalanced datasets

Using SMOTE for class balancing

Evaluating models using ROC-AUC and Precision-Recall

Comparing multiple classification algorithms

Understanding feature importance in tree models

Deploying machine learning models using Streamlit

# Future Improvements

Potential improvements for the system:

Hyperparameter tuning using GridSearchCV

Trying advanced models like XGBoost or LightGBM

Building a real-time fraud detection pipeline

Deploying the model on cloud platforms

# Author

Siya Rathor