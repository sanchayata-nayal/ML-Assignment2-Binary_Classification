import sys
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# --- 1. Import data_prep from parent directory ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_prep import prepare_data

# --- 2. Load Data ---
csv_path = os.path.join(parent_dir, 'framingham_heart_study.csv')
X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(csv_path)

# --- 3. Train Logistic Regression ---
print("Training Logistic Regression...")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# --- 4. Evaluate ---
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")

# --- 5. Save Model ---
# Save dictionary containing model, scaler, and feature names
model_data = {
    'model': log_reg,
    'scaler': scaler,
    'feature_names': feature_names
}

# Save in the same folder as this script
output_path = os.path.join(current_dir, 'logistic_regression_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to {output_path}")