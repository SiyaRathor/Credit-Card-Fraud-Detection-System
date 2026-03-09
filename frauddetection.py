import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATA
# =========================

data = pd.read_csv("Credit-Card-Fraud-Detection-System/creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# SMOTE (FAST VERSION)
# =========================

smote = SMOTE(sampling_strategy=0.5, random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("Dataset after SMOTE:", np.bincount(y_train_smote))

# =========================
# MODEL 1 — Logistic Regression
# =========================

log_model = LogisticRegression(
    max_iter=2000,
    solver="liblinear"
)

log_model.fit(X_train_smote, y_train_smote)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

# Cross validation (reduced folds)
cv_log = cross_val_score(
    log_model,
    X_train_smote,
    y_train_smote,
    cv=3,
    scoring="roc_auc"
)

# =========================
# MODEL 2 — Random Forest (Optimized)
# =========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_smote, y_train_smote)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

cv_rf = cross_val_score(
    rf_model,
    X_train_smote,
    y_train_smote,
    cv=3,
    scoring="roc_auc"
)

# =========================
# MODEL EVALUATION
# =========================

def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n===== {name} =====")
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


evaluate_model("Logistic Regression", y_test, y_pred_log, y_prob_log)
evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

print("\nLogistic CV ROC-AUC:", cv_log.mean())
print("RandomForest CV ROC-AUC:", cv_rf.mean())

# =========================
# ROC CURVE COMPARISON
# =========================

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,5))
plt.plot(fpr_log, tpr_log, label="Logistic")
plt.plot(fpr_rf, tpr_rf, label="RandomForest")
plt.plot([0,1],[0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.show()

# =========================
# PRECISION-RECALL CURVE
# =========================

precision_log, recall_log, _ = precision_recall_curve(y_test, y_prob_log)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

plt.figure(figsize=(6,5))

plt.plot(recall_log, precision_log, label="Logistic")
plt.plot(recall_rf, precision_rf, label="RandomForest")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.legend()
plt.show()

# =========================
# THRESHOLD TUNING
# =========================

threshold = 0.8

y_pred_custom = (y_prob_rf > threshold).astype(int)

print("\n--- RandomForest After Threshold 0.8 ---")
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))

# =========================
# FEATURE IMPORTANCE
# =========================

importances = pd.Series(rf_model.feature_importances_, index=X.columns)

importances.nlargest(10).plot(kind="barh")

plt.title("Top 10 Important Features")
plt.show()

# =========================
# SAVE MODEL FOR DEPLOYMENT
# =========================

joblib.dump(rf_model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\nModel saved successfully!")