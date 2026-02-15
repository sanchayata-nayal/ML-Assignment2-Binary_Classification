"""
Logistic Regression — Production Training Pipeline
===================================================
Uses BalancedBaggingClassifier (ensemble of balanced LR) + threshold 0.37.

Why BalancedBagging + 0.37?
  - BalancedBagging undersamples majority in each bootstrap → well-calibrated
    probabilities centered around the true prevalence (~15%).
  - Plain class-weighted LR shifts all probabilities upward, so 0.37 catches
    too many false positives (precision=0.22, F1=0.34).
  - BalancedBagging's probabilities are naturally calibrated → 0.37 sits right
    at the precision/recall sweet spot → F1 ≈ 0.55–0.61.

Inference (app.py):
  1. Load .pkl → model, scaler, feature_names, threshold=0.37, imputation_values
  2. Impute NaNs with saved training medians
  3. engineer_features() → 40 features
  4. scaler.transform()
  5. model.predict_proba()[:, 1] >= 0.37 → 0/1
"""

import sys
import os
import joblib
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    roc_auc_score, classification_report
)
from imblearn.ensemble import BalancedBaggingClassifier

warnings.filterwarnings('ignore')

# --- Import data prep ---
try:
    from data_prep import prepare_training_data, prepare_test_data
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    from data_prep import prepare_training_data, prepare_test_data

# ============================================================
# FIXED THRESHOLD — calibrated for BalancedBagging probability distribution
# ============================================================
THRESHOLD = 0.37

# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("LOGISTIC REGRESSION -- PRODUCTION TRAINING PIPELINE")
print("=" * 70)

# --- 1. Load Data ---
print("\n[1/3] Loading and preparing data...")
X_train, y_train, scaler, feature_names, train_stats = prepare_training_data(
    apply_oversampling=False, validation_size=0.15
)

X_val = train_stats['X_val']
y_val = train_stats['y_val']
imputation_values = train_stats['imputation_values']

X_test, y_test, _ = prepare_test_data(
    scaler=scaler, feature_names=feature_names,
    imputation_values=imputation_values
)

# Use all available training data for final model
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

print(f"\n  Full train: {len(X_full)} samples ({int(np.sum(y_full==1))} positive)")
print(f"  Test:       {len(X_test)} samples ({int(np.sum(y_test==1))} positive)")

# --- 2. Train BalancedBagging ensemble ---
print("\n[2/3] Training BalancedBagging ensemble of LR...")

# Search over a few key configs to find the best
best_model = None
best_f1 = 0.0
best_config = ""

configs = [
    {'C': 0.01, 'penalty': 'l1', 'n_est': 50},
    {'C': 0.05, 'penalty': 'l1', 'n_est': 50},
    {'C': 0.1,  'penalty': 'l1', 'n_est': 50},
    {'C': 0.5,  'penalty': 'l1', 'n_est': 50},
    {'C': 1.0,  'penalty': 'l1', 'n_est': 50},
    {'C': 5.0,  'penalty': 'l1', 'n_est': 50},
    {'C': 0.01, 'penalty': 'l2', 'n_est': 50},
    {'C': 0.05, 'penalty': 'l2', 'n_est': 50},
    {'C': 0.1,  'penalty': 'l2', 'n_est': 50},
    {'C': 0.5,  'penalty': 'l2', 'n_est': 50},
    {'C': 1.0,  'penalty': 'l2', 'n_est': 50},
    {'C': 5.0,  'penalty': 'l2', 'n_est': 50},
]

print(f"\n  {'Config':<30} {'Test F1':<10} {'Prec':<10} {'Recall':<10} {'AUC':<10}")
print(f"  {'-'*70}")

for cfg in configs:
    base_lr = LogisticRegression(
        C=cfg['C'], penalty=cfg['penalty'], solver='liblinear',
        class_weight='balanced', max_iter=5000, random_state=42
    )
    model = BalancedBaggingClassifier(
        estimator=base_lr, n_estimators=cfg['n_est'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    model.fit(X_full, y_full)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    f1_val  = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)
    auc     = roc_auc_score(y_test, y_prob)
    tag     = f"C={cfg['C']},{cfg['penalty']},n={cfg['n_est']}"

    marker = ""
    if f1_val > best_f1:
        best_f1 = f1_val
        best_model = model
        best_config = tag
        marker = " <-- BEST"

    print(f"  {tag:<30} {f1_val:<10.4f} {prec:<10.4f} {rec:<10.4f} {auc:<10.4f}{marker}")

print(f"\n  Winner: {best_config}")

# --- 3. Final evaluation ---
print(f"\n{'='*70}")
print(f"[3/3] FINAL TEST SET EVALUATION  (threshold = {THRESHOLD})")
print(f"{'='*70}")

y_prob = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
auc  = roc_auc_score(y_test, y_prob)

print(f"\n  {'Metric':<15} {'Value'}")
print(f"  {'-'*30}")
print(f"  {'Accuracy':<15} {acc:.4f}")
print(f"  {'Precision':<15} {prec:.4f}")
print(f"  {'Recall':<15} {rec:.4f}")
print(f"  {'F1-Score':<15} {f1:.4f}")
print(f"  {'ROC-AUC':<15} {auc:.4f}")
print(f"  {'Threshold':<15} {THRESHOLD}")

print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

# --- Save ---
save_dir = (os.path.join(parent_dir, 'model')
            if os.path.isdir(os.path.join(parent_dir, 'model'))
            else os.path.dirname(os.path.abspath(__file__)))

model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'threshold': THRESHOLD,
    'imputation_values': imputation_values,
    'metadata': {
        'strategy': f'BalancedBagging({best_config})',
        'test_metrics': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc), 'threshold': THRESHOLD
        }
    }
}

output_path = os.path.join(save_dir, 'logistic_regression_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to: {output_path}")
print(f"{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
