"""
XGBoost — Production Training Pipeline
=======================================
Compares two XGBoost-based strategies for imbalanced CHD classification:

  Strategy A — XGBoost + scale_pos_weight + early stopping
    - Native gradient reweighting for class imbalance
    - Early stopping on validation prevents overfitting
    - Best n_estimators found automatically (no manual tuning)

  Strategy B — BalancedBaggingClassifier(XGBoost)
    - Each bootstrap bag is balanced (majority undersampled)
    - Smaller XGB per bag → ensemble diversity → robust predictions
    - Well-calibrated probabilities → reliable threshold transfer

F1-optimized threshold tuned on validation set only (no test leakage).
Threshold sweep capped at 0.55 to prevent overshoot when retraining on
full data (retraining shifts probability distribution, inflating thresholds).
Winner retrained on full train+val before final test evaluation.

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
from xgboost import XGBClassifier
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
def find_best_f1_threshold(y_true, y_prob, low=0.20, high=0.55, step=0.005):
    """Sweep thresholds and return best F1 threshold.
    Capped at 0.55 to prevent overshoot after retraining on full data."""
    best_f1, best_t = 0.0, 0.45
    for t in np.arange(low, high + step, step):
        yp = (y_prob >= t).astype(int)
        f = f1_score(y_true, yp, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def find_oracle_threshold(y_true, y_prob, low=0.10, high=0.80, step=0.005):
    """Uncapped sweep for oracle (test-set best possible F1)."""
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
print("XGBOOST -- PRODUCTION TRAINING PIPELINE")
print("=" * 70)

# --- 1. Load Data ---
print("\n[1/5] Loading and preparing data...")
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

spw_train = float(np.sum(y_train == 0)) / float(np.sum(y_train == 1))
spw_full  = float(np.sum(y_full  == 0)) / float(np.sum(y_full  == 1))

print(f"\n  Training:    {len(X_train)} samples ({int(np.sum(y_train==1))} positive)")
print(f"  Validation:  {len(X_val)} samples ({int(np.sum(y_val==1))} positive)")
print(f"  Full train:  {len(X_full)} samples ({int(np.sum(y_full==1))} positive)")
print(f"  Test:        {len(X_test)} samples ({int(np.sum(y_test==1))} positive)")
print(f"  scale_pos_weight (train): {spw_train:.2f}")

all_results = []

# ============================================================
# STRATEGY A: XGBoost + scale_pos_weight + early stopping
# ============================================================
print("\n[2/5] Strategy A: XGBoost + scale_pos_weight + early stopping")
print("  Threshold sweep capped at 0.55 (prevents overshoot after retrain).\n")

configs_a = [
    {'lr': lr, 'depth': d, 'mcw': mcw, 'gamma': g}
    for lr in [0.01, 0.05, 0.1]
    for d in [3, 5, 7]
    for mcw in [1, 3, 5]
    for g in [0, 0.1]
]

print(f"  Searching {len(configs_a)} Strategy-A configs...\n")
print(f"  {'Config':<48} {'Val F1':<8} {'Prec':<8} {'Rec':<8} {'AUC':<8} {'Thr':<7} {'Trees'}")
print(f"  {'-'*95}")

for cfg in configs_a:
    model = XGBClassifier(
        learning_rate=cfg['lr'],
        max_depth=cfg['depth'],
        n_estimators=500,               # max — early stopping will limit
        min_child_weight=cfg['mcw'],
        gamma=cfg['gamma'],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=spw_train,
        eval_metric='logloss',
        early_stopping_rounds=30,
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    best_n_est = model.best_iteration + 1
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_f1_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= threshold).astype(int)

    tag = f"A:lr={cfg['lr']},d={cfg['depth']},mcw={cfg['mcw']},g={cfg['gamma']}"

    result = {
        'strategy': 'A',
        'config': cfg,
        'tag': tag,
        'threshold': threshold,
        'best_n_est': best_n_est,
        'val_f1': val_f1,
        'val_prec': precision_score(y_val, y_pred_val, zero_division=0),
        'val_rec': recall_score(y_val, y_pred_val, zero_division=0),
        'val_auc': roc_auc_score(y_val, y_prob_val),
    }
    all_results.append(result)

    print(f"  {tag:<48} {val_f1:<8.4f} {result['val_prec']:<8.4f} "
          f"{result['val_rec']:<8.4f} {result['val_auc']:<8.4f} {threshold:<7.3f} {best_n_est}")

# ============================================================
# STRATEGY B: BalancedBagging(XGBoost)
# ============================================================
print(f"\n[3/5] Strategy B: BalancedBagging wrapping XGBoost")
print("  Each bag trains on balanced subsample. No scale_pos_weight.\n")

configs_b = [
    {'n_bags': bags, 'xgb_n_est': n, 'xgb_depth': d, 'xgb_lr': lr, 'xgb_mcw': mcw}
    for bags in [30, 50]
    for n in [30, 50]
    for d in [3, 5]
    for lr in [0.05, 0.1]
    for mcw in [1, 3]
]

print(f"  Searching {len(configs_b)} Strategy-B configs...\n")
print(f"  {'Config':<58} {'Val F1':<8} {'Prec':<8} {'Rec':<8} {'AUC':<8} {'Thr':<7}")
print(f"  {'-'*97}")

for cfg in configs_b:
    base_xgb = XGBClassifier(
        learning_rate=cfg['xgb_lr'],
        max_depth=cfg['xgb_depth'],
        n_estimators=cfg['xgb_n_est'],
        min_child_weight=cfg['xgb_mcw'],
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    model = BalancedBaggingClassifier(
        estimator=base_xgb,
        n_estimators=cfg['n_bags'],
        sampling_strategy='auto',
        replacement=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_f1_threshold(y_val, y_prob_val)
    y_pred_val = (y_prob_val >= threshold).astype(int)

    tag = f"B:bags={cfg['n_bags']},n={cfg['xgb_n_est']},d={cfg['xgb_depth']},lr={cfg['xgb_lr']},mcw={cfg['xgb_mcw']}"

    result = {
        'strategy': 'B',
        'config': cfg,
        'tag': tag,
        'threshold': threshold,
        'val_f1': val_f1,
        'val_prec': precision_score(y_val, y_pred_val, zero_division=0),
        'val_rec': recall_score(y_val, y_pred_val, zero_division=0),
        'val_auc': roc_auc_score(y_val, y_prob_val),
    }
    all_results.append(result)

    print(f"  {tag:<58} {val_f1:<8.4f} {result['val_prec']:<8.4f} "
          f"{result['val_rec']:<8.4f} {result['val_auc']:<8.4f} {threshold:<7.3f}")

# ============================================================
# SELECT WINNER
# ============================================================
print(f"\n{'='*70}")
print("[4/5] TOP CONFIGS BY VALIDATION F1 (both strategies)")
print(f"{'='*70}")

top = sorted(all_results, key=lambda x: x['val_f1'], reverse=True)[:15]
print(f"\n  {'Config':<55} {'Val F1':<8} {'Prec':<8} {'Rec':<8} {'Thr':<7}")
print(f"  {'-'*86}")
for i, r in enumerate(top):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {r['tag']:<55} {r['val_f1']:<8.4f} {r['val_prec']:<8.4f} "
          f"{r['val_rec']:<8.4f} {r['threshold']:<7.3f}{marker}")

winner = top[0]
FINAL_THRESHOLD = winner['threshold']
winner_tag = winner['tag']
print(f"\n  Winner: {winner_tag}")
print(f"  Val F1={winner['val_f1']:.4f}, threshold={FINAL_THRESHOLD:.3f}")

# ============================================================
# RETRAIN WINNER ON FULL DATA
# ============================================================
print(f"\n[5/5] Retraining winner on full data ({len(X_full)} samples)...")

if winner['strategy'] == 'A':
    cfg = winner['config']
    n_est = winner['best_n_est']
    print(f"  Strategy A: Using {n_est} trees (from early stopping)")

    final_model = XGBClassifier(
        learning_rate=cfg['lr'],
        max_depth=cfg['depth'],
        n_estimators=n_est,
        min_child_weight=cfg['mcw'],
        gamma=cfg['gamma'],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=spw_full,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model.fit(X_full, y_full)
    strategy_desc = (f"XGB-SPW(lr={cfg['lr']},d={cfg['depth']},"
                     f"mcw={cfg['mcw']},g={cfg['gamma']},n={n_est})")

else:  # Strategy B
    cfg = winner['config']
    print(f"  Strategy B: BalancedBagging({cfg['n_bags']} bags)")

    base_xgb = XGBClassifier(
        learning_rate=cfg['xgb_lr'],
        max_depth=cfg['xgb_depth'],
        n_estimators=cfg['xgb_n_est'],
        min_child_weight=cfg['xgb_mcw'],
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )
    final_model = BalancedBaggingClassifier(
        estimator=base_xgb,
        n_estimators=cfg['n_bags'],
        sampling_strategy='auto',
        replacement=False,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_full, y_full)
    strategy_desc = (f"BB-XGB(bags={cfg['n_bags']},n={cfg['xgb_n_est']},"
                     f"d={cfg['xgb_depth']},lr={cfg['xgb_lr']},mcw={cfg['xgb_mcw']})")

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

oracle_t, oracle_f1 = find_oracle_threshold(y_test, y_prob)
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
        'strategy': strategy_desc,
        'winner_strategy': winner['strategy'],
        'val_f1': float(winner['val_f1']),
        'test_metrics': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc), 'threshold': float(FINAL_THRESHOLD)
        }
    }
}

output_path = os.path.join(save_dir, 'xgboost_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to: {output_path}")
print(f"{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")