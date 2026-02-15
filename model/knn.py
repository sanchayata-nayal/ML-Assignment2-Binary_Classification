"""
k-Nearest Neighbors — Production Training Pipeline
====================================================
Compares seven KNN-based strategies for imbalanced classification:

  Raw-feature strategies (40 dims):
    A. SMOTE + KNN          — oversample minority, then standard KNN
    B. SMOTEENN + KNN       — SMOTE followed by Edited-NN cleaning
    C. SMOTETomek + KNN     — SMOTE followed by Tomek-link removal
    D. BalancedBagging KNN  — ensemble of KNN on balanced subsamples

  Dimensionality-reduction strategies (combat curse of dimensionality):
    E. PCA + SMOTEENN + KNN       — PCA reduces feature space, then SMOTEENN + KNN
    F. PCA + BalancedBagging KNN  — PCA inside each bagging estimator
    G. SelectKBest + SMOTEENN + KNN — mutual-info feature selection + SMOTEENN + KNN

KNN is distance-based → highly sensitive to irrelevant features.
Reducing 40 → ~10-20 informative dimensions makes distances more
discriminative and typically yields large F1 improvements.

F1-optimized threshold is tuned on validation set only (no test leakage).
Winner is retrained on full train+val before final test evaluation.
Pipeline objects are used so that PCA/SelectKBest transforms are
applied automatically at inference time (no app.py changes needed).

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    roc_auc_score, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
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
def find_best_f1_threshold(y_true, y_prob, low=0.05, high=0.50, step=0.005):
    """Sweep thresholds and return the one that maximizes F1.

    High threshold cap (0.50) is intentional: KNN trained on SMOTE/SMOTEENN
    data outputs inflated probabilities because the resampled training set
    has ~50% or more positives, but real-world prevalence is ~15%.
    Allowing thresholds above 0.50 leads to overfitting on validation and
    poor generalization — restrict to a range that reflects deployment.
    """
    best_f1, best_t = 0.0, 0.35
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
print("k-NEAREST NEIGHBORS -- PRODUCTION TRAINING PIPELINE")
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
print(f"  Features:    {X_train.shape[1]} (raw)")

# --- 2. Strategy search (all threshold tuning on VALIDATION only) ---
print("\n[2/4] Running multi-strategy KNN search...")

all_results = []

# Common KNN hyperparameter grid (raw-feature strategies)
knn_params = [
    {'k': k, 'w': w, 'p': p}
    for k in [3, 5, 7, 9, 11, 15, 21]
    for w in ['uniform', 'distance']
    for p in [1, 2]              # 1 = Manhattan, 2 = Euclidean
]

# Reduced grid for dim-reduction strategies (more outer combos)
knn_params_reduced = [
    {'k': k, 'w': w, 'p': p}
    for k in [3, 5, 7, 11, 15, 21]
    for w in ['uniform', 'distance']
    for p in [1, 2]
]


# ================================================================
# RAW-FEATURE STRATEGIES (A–D)
# ================================================================

# ---- Strategy A: SMOTE + KNN ----
print("\n  === Strategy A: SMOTE + KNN ===")
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"  SMOTE resampled: {len(X_smote)} samples "
      f"(0: {int(np.sum(y_smote==0))}, 1: {int(np.sum(y_smote==1))})")
print(f"  Evaluating {len(knn_params)} configs...")

for cfg in knn_params:
    model = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    name = f"SMOTE-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']})"
    r = evaluate_on_val(name, model, X_smote, y_smote, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'SMOTE'
    all_results.append(r)

smote_best = max([r for r in all_results if r['strategy'] == 'SMOTE'],
                 key=lambda x: x['val_f1'])
print(f"  Best SMOTE-KNN: {smote_best['name']} --> val F1={smote_best['val_f1']:.4f}")

# ---- Strategy B: SMOTEENN + KNN ----
print("\n  === Strategy B: SMOTEENN + KNN ===")
smoteenn = SMOTEENN(random_state=42)
X_se, y_se = smoteenn.fit_resample(X_train, y_train)
print(f"  SMOTEENN resampled: {len(X_se)} samples "
      f"(0: {int(np.sum(y_se==0))}, 1: {int(np.sum(y_se==1))})")
print(f"  Evaluating {len(knn_params)} configs...")

for cfg in knn_params:
    model = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    name = f"SE-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']})"
    r = evaluate_on_val(name, model, X_se, y_se, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'SMOTEENN'
    all_results.append(r)

se_best = max([r for r in all_results if r['strategy'] == 'SMOTEENN'],
              key=lambda x: x['val_f1'])
print(f"  Best SMOTEENN-KNN: {se_best['name']} --> val F1={se_best['val_f1']:.4f}")

# ---- Strategy C: SMOTETomek + KNN ----
print("\n  === Strategy C: SMOTETomek + KNN ===")
smotetomek = SMOTETomek(random_state=42)
X_st, y_st = smotetomek.fit_resample(X_train, y_train)
print(f"  SMOTETomek resampled: {len(X_st)} samples "
      f"(0: {int(np.sum(y_st==0))}, 1: {int(np.sum(y_st==1))})")
print(f"  Evaluating {len(knn_params)} configs...")

for cfg in knn_params:
    model = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    name = f"ST-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']})"
    r = evaluate_on_val(name, model, X_st, y_st, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'SMOTETomek'
    all_results.append(r)

st_best = max([r for r in all_results if r['strategy'] == 'SMOTETomek'],
              key=lambda x: x['val_f1'])
print(f"  Best SMOTETomek-KNN: {st_best['name']} --> val F1={st_best['val_f1']:.4f}")

# ---- Strategy D: BalancedBagging KNN ----
print("\n  === Strategy D: BalancedBagging KNN ===")
bb_knn_params = [
    {'k': k, 'w': w, 'p': p, 'n': n}
    for k in [3, 5, 7, 11]
    for w in ['uniform', 'distance']
    for p in [1, 2]
    for n in [30, 50, 80]
]
print(f"  Evaluating {len(bb_knn_params)} configs...")

for cfg in bb_knn_params:
    base = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=1
    )
    model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    name = f"BB-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']},n={cfg['n']})"
    r = evaluate_on_val(name, model, X_train, y_train, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'BalancedBagging'
    all_results.append(r)

bb_best = max([r for r in all_results if r['strategy'] == 'BalancedBagging'],
              key=lambda x: x['val_f1'])
print(f"  Best BB-KNN: {bb_best['name']} --> val F1={bb_best['val_f1']:.4f}")


# ================================================================
# DIMENSIONALITY-REDUCTION STRATEGIES (E–G)
# KNN is distance-based → curse of dimensionality with 40 features.
# Reducing to ~10-20 informative dims makes distances meaningful.
# ================================================================

# ---- Strategy E: PCA + SMOTEENN + KNN ----
print("\n  === Strategy E: PCA + SMOTEENN + KNN ===")
pca_components = [8, 10, 12, 15, 20, 25]
total_e = len(pca_components) * len(knn_params_reduced)
print(f"  Evaluating {total_e} configs ({len(pca_components)} PCA dims x {len(knn_params_reduced)} KNN)...")

for n_comp in pca_components:
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    se = SMOTEENN(random_state=42)
    X_se_pca, y_se_pca = se.fit_resample(X_train_pca, y_train)

    for cfg in knn_params_reduced:
        knn = KNeighborsClassifier(
            n_neighbors=cfg['k'], weights=cfg['w'],
            metric='minkowski', p=cfg['p'], n_jobs=-1
        )
        name = f"PCA{n_comp}-SE-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']})"
        r = evaluate_on_val(name, knn, X_se_pca, y_se_pca, X_val_pca, y_val)
        r['config'] = {**cfg, 'n_components': n_comp}
        r['strategy'] = 'PCA_SMOTEENN'
        all_results.append(r)

pca_se_best = max([r for r in all_results if r['strategy'] == 'PCA_SMOTEENN'],
                   key=lambda x: x['val_f1'])
print(f"  Best PCA+SE-KNN: {pca_se_best['name']} --> val F1={pca_se_best['val_f1']:.4f}")

# ---- Strategy F: PCA + BalancedBagging KNN ----
# PCA is fitted *inside* each bag via Pipeline → fully automatic at inference
print("\n  === Strategy F: PCA + BalancedBagging KNN ===")
bb_pca_params = [
    {'k': k, 'w': w, 'p': p, 'n': n, 'n_components': nc}
    for nc in [10, 15, 20]
    for k in [3, 5, 7, 11]
    for w in ['uniform', 'distance']
    for p in [1, 2]
    for n in [30, 50]
]
print(f"  Evaluating {len(bb_pca_params)} configs...")

for cfg in bb_pca_params:
    base = Pipeline([
        ('pca', PCA(n_components=cfg['n_components'], random_state=42)),
        ('knn', KNeighborsClassifier(
            n_neighbors=cfg['k'], weights=cfg['w'],
            metric='minkowski', p=cfg['p'], n_jobs=1
        ))
    ])
    model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    name = f"BB-PCA{cfg['n_components']}-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']},n={cfg['n']})"
    r = evaluate_on_val(name, model, X_train, y_train, X_val, y_val)
    r['config'] = cfg
    r['strategy'] = 'BB_PCA'
    all_results.append(r)

bb_pca_best = max([r for r in all_results if r['strategy'] == 'BB_PCA'],
                   key=lambda x: x['val_f1'])
print(f"  Best BB-PCA-KNN: {bb_pca_best['name']} --> val F1={bb_pca_best['val_f1']:.4f}")

# ---- Strategy G: SelectKBest + SMOTEENN + KNN ----
print("\n  === Strategy G: SelectKBest (MI) + SMOTEENN + KNN ===")
k_features_list = [8, 10, 12, 15, 20]
total_g = len(k_features_list) * len(knn_params_reduced)
print(f"  Evaluating {total_g} configs ({len(k_features_list)} feature counts x {len(knn_params_reduced)} KNN)...")

for k_feat in k_features_list:
    selector = SelectKBest(mutual_info_classif, k=k_feat)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)

    se = SMOTEENN(random_state=42)
    X_se_sel, y_se_sel = se.fit_resample(X_train_sel, y_train)

    for cfg in knn_params_reduced:
        knn = KNeighborsClassifier(
            n_neighbors=cfg['k'], weights=cfg['w'],
            metric='minkowski', p=cfg['p'], n_jobs=-1
        )
        name = f"SK{k_feat}-SE-KNN(k={cfg['k']},w={cfg['w']},p={cfg['p']})"
        r = evaluate_on_val(name, knn, X_se_sel, y_se_sel, X_val_sel, y_val)
        r['config'] = {**cfg, 'k_features': k_feat}
        r['strategy'] = 'SelectK_SMOTEENN'
        all_results.append(r)

sk_best = max([r for r in all_results if r['strategy'] == 'SelectK_SMOTEENN'],
               key=lambda x: x['val_f1'])
print(f"  Best SelectK+SE-KNN: {sk_best['name']} --> val F1={sk_best['val_f1']:.4f}")


# --- 3. Select overall winner (by val F1) ---
print(f"\n{'='*70}")
print("[3/4] STRATEGY COMPARISON (top 20 by val F1)")
print(f"{'='*70}")

top = sorted(all_results, key=lambda x: x['val_f1'], reverse=True)[:20]
print(f"\n  {'Name':<55} {'Val F1':<9} {'Prec':<9} {'Rec':<9} {'AUC':<9} {'Thr':<8}")
print(f"  {'-'*99}")
for i, r in enumerate(top):
    marker = " <-- BEST" if i == 0 else ""
    print(f"  {r['name']:<55} {r['val_f1']:<9.4f} {r['val_prec']:<9.4f} "
          f"{r['val_rec']:<9.4f} {r['val_auc']:<9.4f} {r['threshold']:<8.3f}{marker}")

winner = top[0]
print(f"\n  Winner: {winner['name']} (val F1={winner['val_f1']:.4f}, threshold={winner['threshold']:.3f})")

# --- 4. Retrain winner on full data → final test evaluation ---
print(f"\n[4/4] Retraining winner on full data ({len(X_full)} samples)...")

cfg = winner['config']
strategy = winner['strategy']

# --- Build final model based on winning strategy ---
if strategy in ('SMOTE', 'SMOTEENN', 'SMOTETomek'):
    # Raw-feature resampling strategies
    if strategy == 'SMOTE':
        resampler = SMOTE(random_state=42, k_neighbors=5)
    elif strategy == 'SMOTEENN':
        resampler = SMOTEENN(random_state=42)
    else:
        resampler = SMOTETomek(random_state=42)
    X_resampled, y_resampled = resampler.fit_resample(X_full, y_full)
    print(f"  {strategy} on full data: {len(X_resampled)} samples "
          f"(0: {int(np.sum(y_resampled==0))}, 1: {int(np.sum(y_resampled==1))})")
    final_model = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    final_model.fit(X_resampled, y_resampled)

elif strategy == 'BalancedBagging':
    # Raw-feature BalancedBagging
    base = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=1
    )
    final_model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_full, y_full)

elif strategy == 'PCA_SMOTEENN':
    # PCA + SMOTEENN: fit PCA on full data, resample, train KNN, wrap in Pipeline
    n_comp = cfg['n_components']
    pca_final = PCA(n_components=n_comp, random_state=42)
    X_full_pca = pca_final.fit_transform(X_full)
    se_final = SMOTEENN(random_state=42)
    X_resampled, y_resampled = se_final.fit_resample(X_full_pca, y_full)
    print(f"  PCA({n_comp}) + SMOTEENN on full data: {len(X_resampled)} samples "
          f"(0: {int(np.sum(y_resampled==0))}, 1: {int(np.sum(y_resampled==1))})")
    knn_final = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    knn_final.fit(X_resampled, y_resampled)
    # Wrap in Pipeline so model.predict_proba() applies PCA transform automatically
    final_model = Pipeline([('pca', pca_final), ('knn', knn_final)])

elif strategy == 'BB_PCA':
    # BalancedBagging with Pipeline(PCA + KNN) base — PCA fitted per bag
    base = Pipeline([
        ('pca', PCA(n_components=cfg['n_components'], random_state=42)),
        ('knn', KNeighborsClassifier(
            n_neighbors=cfg['k'], weights=cfg['w'],
            metric='minkowski', p=cfg['p'], n_jobs=1
        ))
    ])
    final_model = BalancedBaggingClassifier(
        estimator=base, n_estimators=cfg['n'],
        sampling_strategy='auto', replacement=False,
        random_state=42, n_jobs=-1
    )
    final_model.fit(X_full, y_full)

elif strategy == 'SelectK_SMOTEENN':
    # SelectKBest + SMOTEENN: fit selector on full data, resample, train KNN, wrap in Pipeline
    k_feat = cfg['k_features']
    selector_final = SelectKBest(mutual_info_classif, k=k_feat)
    X_full_sel = selector_final.fit_transform(X_full, y_full)
    se_final = SMOTEENN(random_state=42)
    X_resampled, y_resampled = se_final.fit_resample(X_full_sel, y_full)
    print(f"  SelectKBest({k_feat}) + SMOTEENN on full data: {len(X_resampled)} samples "
          f"(0: {int(np.sum(y_resampled==0))}, 1: {int(np.sum(y_resampled==1))})")
    knn_final = KNeighborsClassifier(
        n_neighbors=cfg['k'], weights=cfg['w'],
        metric='minkowski', p=cfg['p'], n_jobs=-1
    )
    knn_final.fit(X_resampled, y_resampled)
    # Wrap in Pipeline so model.predict_proba() applies feature selection automatically
    final_model = Pipeline([('select', selector_final), ('knn', knn_final)])

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
        'best_config': {k: v for k, v in cfg.items()},
        'val_f1': float(winner['val_f1']),
        'test_metrics': {
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1),
            'roc_auc': float(auc), 'threshold': float(FINAL_THRESHOLD)
        }
    }
}

output_path = os.path.join(save_dir, 'knn_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to: {output_path}")
print(f"{'='*70}")
print("TRAINING COMPLETE")
print(f"{'='*70}")