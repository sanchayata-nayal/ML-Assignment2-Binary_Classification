"""
Random Forest — Production Training Pipeline
=============================================
Uses BalancedBaggingClassifier wrapping DecisionTreeClassifier(max_features='sqrt')
— functionally a Balanced Random Forest — with F1-optimized threshold tuned on
a held-out validation set.

Why BalancedBagging(DT) instead of plain RF?
  - Plain RF with class_weight='balanced' shifts all probabilities upward,
    making threshold selection fragile.
  - BalancedBagging undersamples majority in each bootstrap → each tree
    sees a balanced dataset → well-calibrated probabilities.
  - max_features='sqrt' on the base DT provides the same feature subsampling
    as a standard Random Forest (random subspace at each split).

Why threshold sweep (not fixed 0.37)?
  - Tree-based BalancedBagging produces different probability distributions
    than LR-based BalancedBagging.
  - A fixed 0.37 is too aggressive → precision=0.22, massive false positives.
  - Sweeping on validation finds the precision/recall sweet spot for RF.

Grid searches over max_depth, min_samples_leaf, n_estimators.
Winner retrained on full train+val.

Inference (app.py):
  1. Load .pkl → model, scaler, feature_names, threshold, imputation_values
  2. Impute NaNs with saved training medians
  3. engineer_features() → 40 features
  4. scaler.transform()
  5. model.predict_proba()[:, 1] >= threshold → 0/1
"""

import sys
import os
import joblib
import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
# UTILITY
# ============================================================
def find_best_f1_threshold(y_true, y_prob, low=0.30, high=0.70, step=0.005):
    """Sweep thresholds and return best F1 threshold."""
    best_f1, best_t = 0.0, 0.50
    for t in np.arange(low, high, step):
        yp = (y_prob >= t).astype(int)
        f = f1_score(y_true, yp, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("RANDOM FOREST -- PRODUCTION TRAINING PIPELINE")
print("=" * 70)

# --- 1. Load Data ---
print("\n[1/4] Loading and preparing data...")
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

X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

print(f"\n  Training:    {len(X_train)} samples ({int(np.sum(y_train==1))} positive)")
print(f"  Validation:  {len(X_val)} samples ({int(np.sum(y_val==1))} positive)")
print(f"  Full train:  {len(X_full)} samples ({int(np.sum(y_full==1))} positive)")
print(f"  Test:        {len(X_test)} samples ({int(np.sum(y_test==1))} positive)")

# --- 2. Grid search — train on X_train, tune threshold on X_val ---
print("\n[2/4] Training BalancedBagging ensemble of DecisionTree (Balanced RF)...")
print("  Threshold tuned on VALIDATION set (no test leakage).\n")

configs = [
    {'max_depth': d, 'min_samples_leaf': l, 'n_est': n}
    for d in [5, 8, 10, 15, 20, None]
    for l in [1, 3, 5, 8]
    for n in [100, 200]
]

all_results = []

print(f"  Searching {len(configs)} configurations...\n")
print(f"  {'Config':<30} {'Val F1':<9} {'Prec':<9} {'Rec':<9} {'AUC':<9} {'Thr':<8}")
print(f"  {'-'*74}")

for cfg in configs:
    base_dt = DecisionTreeClassifier(
        max_depth=cfg['max_depth'],
        min_samples_leaf=cfg['min_samples_leaf'],
        max_features='sqrt',
        random_state=42
    )
    model = BalancedBaggingClassifier(
        estimator=base_dt, n_estimators=cfg['n_est'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_f1_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= threshold).astype(int)

    depth_str = str(cfg['max_depth']) if cfg['max_depth'] is not None else 'None'
    tag = f"d={depth_str},leaf={cfg['min_samples_leaf']},n={cfg['n_est']}"

    result = {
        'config': cfg,
        'tag': tag,
        'threshold': threshold,
        'val_f1': val_f1,
        'val_prec': precision_score(y_val, y_pred_val, zero_division=0),
        'val_rec': recall_score(y_val, y_pred_val, zero_division=0),
        'val_auc': roc_auc_score(y_val, y_prob_val),
    }
    all_results.append(result)

    print(f"  {tag:<30} {val_f1:<9.4f} {result['val_prec']:<9.4f} "
          f"{result['val_rec']:<9.4f} {result['val_auc']:<9.4f} {threshold:<8.3f}")

# --- 3. Select winner & show top configs ---
print(f"\n{'='*70}")
print("[3/4] TOP CONFIGS BY VALIDATION F1")
print(f"{'='*70}")

top = sorted(all_results, key=lambda x: x['val_f1'], reverse=True)[:10]
print(f"\n  {'Config':<30} {'Val F1':<9} {'Prec':<9} {'Rec':<9} {'AUC':<9} {'Thr':<8}")
print(f"  {'-'*74}")
for i, r in enumerate(top):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {r['tag']:<30} {r['val_f1']:<9.4f} {r['val_prec']:<9.4f} "
          f"{r['val_rec']:<9.4f} {r['val_auc']:<9.4f} {r['threshold']:<8.3f}{marker}")

winner = top[0]
winner_cfg = winner['config']
FINAL_THRESHOLD = winner['threshold']
winner_tag = winner['tag']
print(f"\n  Winner: {winner_tag} (val F1={winner['val_f1']:.4f}, threshold={FINAL_THRESHOLD:.3f})")

# --- 4. Retrain winner on full data → final test evaluation ---
print(f"\n[4/4] Retraining winner on full data ({len(X_full)} samples)...")

final_model = BalancedBaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=winner_cfg['max_depth'],
        min_samples_leaf=winner_cfg['min_samples_leaf'],
        max_features='sqrt',
        random_state=42
    ),
    n_estimators=winner_cfg['n_est'],
    sampling_strategy='auto', replacement=False,
    random_state=42, n_jobs=-1
)
final_model.fit(X_full, y_full)

# --- Final test set evaluation ---
print(f"\n{'='*70}")
print(f"FINAL TEST SET EVALUATION  (threshold = {FINAL_THRESHOLD:.4f})")
print(f"{'='*70}")

y_prob = final_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= FINAL_THRESHOLD).astype(int)

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
print(f"  {'Threshold':<15} {FINAL_THRESHOLD:.4f}")

print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

oracle_t, oracle_f1 = find_best_f1_threshold(y_test, y_prob)
print(f"  [Oracle] Best possible test F1: {oracle_f1:.4f} at threshold {oracle_t:.4f}")

# --- Save ---
save_dir = (os.path.join(parent_dir, 'model')
            if os.path.isdir(os.path.join(parent_dir, 'model'))
            else os.path.dirname(os.path.abspath(__file__)))

model_data = {
    'model': final_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'threshold': FINAL_THRESHOLD,
    'imputation_values': imputation_values,
    'metadata': {
        'strategy': f'BalancedBagging-RF({winner_tag})',
        'best_config': winner_cfg,
        'val_f1': float(winner['val_f1']),
        'test_metrics': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc), 'threshold': float(FINAL_THRESHOLD)
        }
    }
}

output_path = os.path.join(save_dir, 'random_forest_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to: {output_path}")
print(f"{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")