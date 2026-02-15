"""
Decision Tree — Production Training Pipeline
=============================================
Compares three DT-based ensemble strategies for imbalanced classification:
  1. BalancedBaggingClassifier — bagging with per-bag majority undersampling
  2. EasyEnsembleClassifier — ensemble of AdaBoost on balanced subsamples
     (AdaBoost produces smoother, better-calibrated probabilities than bagging)
  3. RUSBoostClassifier — random undersampling + AdaBoost boosting

All are fundamentally Decision-Tree ensembles. F1-optimized threshold is
tuned on validation set only (no test leakage). Winner is retrained on
full train+val before final test evaluation.

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
from imblearn.ensemble import (
    BalancedBaggingClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier
)

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
def find_best_f1_threshold(y_true, y_prob, low=0.05, high=0.95, step=0.005):
    """Sweep thresholds and return the one that maximizes F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(low, high, step):
        yp = (y_prob >= t).astype(int)
        f = f1_score(y_true, yp, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def evaluate_on_val(name, model, X_train, y_train, X_val, y_val):
    """Train on X_train, tune threshold on X_val via F1. Return dict."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_f1_threshold(y_val, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'name': name,
        'model': model,
        'threshold': threshold,
        'val_f1': val_f1,
        'val_prec': precision_score(y_val, y_pred, zero_division=0),
        'val_rec': recall_score(y_val, y_pred, zero_division=0),
        'val_auc': roc_auc_score(y_val, y_prob),
    }


# ============================================================
# MAIN
# ============================================================
print("=" * 70)
print("DECISION TREE -- PRODUCTION TRAINING PIPELINE")
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

# --- 2. Strategy search (all threshold tuning on VALIDATION only) ---
print("\n[2/4] Running multi-strategy DT ensemble search...")

all_results = []

# ---- Strategy A: BalancedBagging DT ----
print("\n  === Strategy A: BalancedBagging DT ===")
bb_configs = [
    {'depth': d, 'leaf': l, 'feat': f, 'n': n}
    for d in [5, 6, 8, 10, None]
    for l in [3, 5, 10]
    for f in ['sqrt', 0.7, None]
    for n in [100, 200]
]
print(f"  Evaluating {len(bb_configs)} configs...")

for cfg in bb_configs:
    base = DecisionTreeClassifier(
        max_depth=cfg['depth'], min_samples_split=10,
        min_samples_leaf=cfg['leaf'], max_features=cfg['feat'],
        random_state=42
    )
    model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    d_s = str(cfg['depth']) if cfg['depth'] else 'None'
    f_s = str(cfg['feat']) if cfg['feat'] else 'all'
    name = f"BB(d={d_s},l={cfg['leaf']},f={f_s},n={cfg['n']})"
    r = evaluate_on_val(name, model, X_train, y_train, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'BalancedBagging'
    all_results.append(r)

bb_best = max([r for r in all_results if r['strategy'] == 'BalancedBagging'],
              key=lambda x: x['val_f1'])
print(f"  Best BB: {bb_best['name']} --> val F1={bb_best['val_f1']:.4f}")

# ---- Strategy B: EasyEnsemble (AdaBoost-based) ----
print("\n  === Strategy B: EasyEnsembleClassifier ===")
ee_configs = [
    {'n': n}
    for n in [10, 20, 30, 50, 80]
]
print(f"  Evaluating {len(ee_configs)} configs...")

for cfg in ee_configs:
    model = EasyEnsembleClassifier(
        n_estimators=cfg['n'],
        sampling_strategy='auto',
        random_state=42, n_jobs=-1
    )
    name = f"EE(n={cfg['n']})"
    r = evaluate_on_val(name, model, X_train, y_train, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'EasyEnsemble'
    all_results.append(r)

ee_best = max([r for r in all_results if r['strategy'] == 'EasyEnsemble'],
              key=lambda x: x['val_f1'])
print(f"  Best EE: {ee_best['name']} --> val F1={ee_best['val_f1']:.4f}")

# ---- Strategy C: RUSBoost ----
print("\n  === Strategy C: RUSBoostClassifier ===")
rus_configs = [
    {'n': n, 'lr': lr, 'depth': d}
    for n in [50, 100, 200, 500]
    for lr in [0.1, 0.5, 1.0]
    for d in [1, 2, 3]
]
print(f"  Evaluating {len(rus_configs)} configs...")

for cfg in rus_configs:
    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=42)
    model = RUSBoostClassifier(
        estimator=base,
        n_estimators=cfg['n'],
        learning_rate=cfg['lr'],
        sampling_strategy='auto',
        random_state=42
    )
    name = f"RUS(n={cfg['n']},lr={cfg['lr']},d={cfg['depth']})"
    r = evaluate_on_val(name, model, X_train, y_train, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'RUSBoost'
    all_results.append(r)

rus_best = max([r for r in all_results if r['strategy'] == 'RUSBoost'],
               key=lambda x: x['val_f1'])
print(f"  Best RUS: {rus_best['name']} --> val F1={rus_best['val_f1']:.4f}")

# --- 3. Select overall winner (by val F1) ---
print(f"\n{'='*70}")
print("[3/4] STRATEGY COMPARISON (top 15 by val F1)")
print(f"{'='*70}")

top = sorted(all_results, key=lambda x: x['val_f1'], reverse=True)[:15]
print(f"\n  {'Name':<40} {'Val F1':<9} {'Prec':<9} {'Rec':<9} {'AUC':<9} {'Thr':<8}")
print(f"  {'-'*84}")
for i, r in enumerate(top):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {r['name']:<40} {r['val_f1']:<9.4f} {r['val_prec']:<9.4f} "
          f"{r['val_rec']:<9.4f} {r['val_auc']:<9.4f} {r['threshold']:<8.3f}{marker}")

winner = top[0]
print(f"\n  Winner: {winner['name']} (val F1={winner['val_f1']:.4f}, threshold={winner['threshold']:.3f})")

# --- 4. Retrain winner on full data → final test evaluation ---
print(f"\n[4/4] Retraining winner on full data ({len(X_full)} samples)...")

# Rebuild the winner model from its config
cfg = winner['config']
strategy = winner['strategy']

if strategy == 'BalancedBagging':
    base = DecisionTreeClassifier(
        max_depth=cfg['depth'], min_samples_split=10,
        min_samples_leaf=cfg['leaf'], max_features=cfg['feat'],
        random_state=42
    )
    final_model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
elif strategy == 'EasyEnsemble':
    final_model = EasyEnsembleClassifier(
        n_estimators=cfg['n'],
        sampling_strategy='auto',
        random_state=42, n_jobs=-1
    )
elif strategy == 'RUSBoost':
    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=42)
    final_model = RUSBoostClassifier(
        estimator=base,
        n_estimators=cfg['n'],
        learning_rate=cfg['lr'],
        sampling_strategy='auto',
        random_state=42
    )

final_model.fit(X_full, y_full)
FINAL_THRESHOLD = winner['threshold']

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
        'strategy': f'{strategy}({winner["name"]})',
        'best_config': cfg,
        'val_f1': float(winner['val_f1']),
        'test_metrics': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc), 'threshold': float(FINAL_THRESHOLD)
        }
    }
}

output_path = os.path.join(save_dir, 'decision_tree_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to: {output_path}")
print(f"{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")
