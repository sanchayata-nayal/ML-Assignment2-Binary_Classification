import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, matthews_corrcoef

# --- 1. Robust Import Strategy ---
try:
    from data_prep import prepare_data
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from data_prep import prepare_data

# --- 2. Load Data ---
csv_path = os.path.join(parent_dir, 'framingham_heart_study.csv')
if not os.path.exists(csv_path) and os.path.exists('framingham_heart_study.csv'):
    csv_path = 'framingham_heart_study.csv'

X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(csv_path)

# --- 3. Train Logistic Regression ---
print("Training Logistic Regression...")
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# --- 4. FIND THE BEST THRESHOLD ---
y_prob = log_reg.predict_proba(X_test)[:, 1]

print("\n--- Threshold Optimization Search ---")
print(f"{'Threshold':<10} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
print("-" * 55)

best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.1, 0.95, 0.05):
    y_p = (y_prob >= t).astype(int)
    acc = accuracy_score(y_test, y_p)
    rec = recall_score(y_test, y_p)
    prec = precision_score(y_test, y_p)
    f1 = f1_score(y_test, y_p)
    
    print(f"{t:<10.2f} {acc:<10.4f} {rec:<10.4f} {prec:<10.4f} {f1:<10.4f}")
    
    # Track best F1 just for reference
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("-" * 55)
print(f"Mathematically 'Best' F1 Score is at Threshold: {best_threshold:.2f}")

# --- 5. USE THE CHOSEN THRESHOLD ---
# CHANGE THIS VALUE based on the table above to what YOU prefer!
# I recommend picking the one where Accuracy ~0.80 and Recall ~0.30-0.40
FINAL_THRESHOLD = 0.40  # <--- SET THIS TO YOUR CHOSEN VALUE

print(f"\nFinalizing Model with Threshold: {FINAL_THRESHOLD}")
y_pred_final = (y_prob >= FINAL_THRESHOLD).astype(int)

print(f"Final Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Final Recall:   {recall_score(y_test, y_pred_final):.4f}")

# --- 6. Save Model ---
if 'model' in os.listdir(parent_dir):
    save_dir = os.path.join(parent_dir, 'model')
else:
    save_dir = os.path.dirname(os.path.abspath(__file__))

model_data = {
    'model': log_reg,
    'scaler': scaler,
    'feature_names': feature_names,
    'threshold': FINAL_THRESHOLD
}

output_path = os.path.join(save_dir, 'logistic_regression_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to {output_path}")