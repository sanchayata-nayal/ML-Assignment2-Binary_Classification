"""
Production-Level Training Orchestrator
=======================================
Runs all 6 production-grade model training scripts sequentially and
collects their metrics into a unified comparison report.

Each model script (model/*.py) contains its own sophisticated training
pipeline with:
  - BalancedBagging / SMOTE / ensemble strategies
  - Validation-based F1 threshold optimization
  - Grid search over model-specific hyperparameters
  - Proper train/val/test split (no test leakage)

This orchestrator:
  1. Executes each script as a subprocess
  2. Loads the saved .pkl artifacts to extract metrics
  3. Prints a unified comparison table
  4. Saves model/training_report.json
"""

import os
import sys
import subprocess
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime


# ============================================================
# MODEL REGISTRY — maps to individual training scripts
# ============================================================

MODEL_SCRIPTS = [
    {
        'key': 'logistic_regression',
        'name': 'Logistic Regression',
        'script': 'model/logistic_regression.py',
        'pkl': 'logistic_regression_model.pkl',
    },
    {
        'key': 'decision_tree',
        'name': 'Decision Tree',
        'script': 'model/decision_tree.py',
        'pkl': 'decision_tree_model.pkl',
    },
    {
        'key': 'knn',
        'name': 'k-Nearest Neighbors',
        'script': 'model/knn.py',
        'pkl': 'knn_model.pkl',
    },
    {
        'key': 'naive_bayes',
        'name': 'Naive Bayes',
        'script': 'model/naive_bayes.py',
        'pkl': 'naive_bayes_model.pkl',
    },
    {
        'key': 'random_forest',
        'name': 'Random Forest',
        'script': 'model/random_forest.py',
        'pkl': 'random_forest_model.pkl',
    },
    {
        'key': 'xgboost',
        'name': 'XGBoost',
        'script': 'model/xgboost_model.py',
        'pkl': 'xgboost_model.pkl',
    },
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def run_model_script(script_path):
    """
    Run an individual model training script as a subprocess.

    Args:
        script_path: Relative path to the Python training script.

    Returns:
        tuple: (return_code, stdout, stderr)
    """
    result = subprocess.run(
        [sys.executable, '-u', script_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return result.returncode, result.stdout, result.stderr


def load_model_metrics(pkl_filename):
    """
    Load a trained .pkl artifact and extract test metrics.

    Args:
        pkl_filename: Name of the .pkl file inside model/ directory.

    Returns:
        dict: Extracted metrics with keys: accuracy, precision, recall, f1,
              roc_auc, threshold, strategy.
    """
    pkl_path = os.path.join('model', pkl_filename)
    if not os.path.exists(pkl_path):
        return None

    artifacts = joblib.load(pkl_path)
    metadata = artifacts.get('metadata', {})
    test_metrics = metadata.get('test_metrics', {})

    return {
        'threshold': artifacts.get('threshold', test_metrics.get('threshold', 0.5)),
        'accuracy': test_metrics.get('accuracy', 0.0),
        'precision': test_metrics.get('precision', 0.0),
        'recall': test_metrics.get('recall', 0.0),
        'f1': test_metrics.get('f1', 0.0),
        'roc_auc': test_metrics.get('roc_auc', 0.0),
        'strategy': metadata.get('strategy', 'N/A'),
    }


def load_trained_model(model_key):
    """
    Load a trained model with its artifacts for inference.

    Args:
        model_key: Model identifier (e.g., 'random_forest').

    Returns:
        dict: Contains model, scaler, feature_names, threshold, etc.
    """
    model_path = f'model/{model_key}_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_with_model(model_key, X_data):
    """
    Make predictions using a trained model with optimized threshold.

    Args:
        model_key: Model identifier.
        X_data: Input features (numpy array or DataFrame).

    Returns:
        dict: predictions, probabilities, threshold, model_name.
    """
    artifacts = load_trained_model(model_key)
    model = artifacts['model']
    threshold = artifacts.get('threshold', 0.5)

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        predictions = model.predict(X_data)
        probabilities = None

    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'threshold': threshold,
        'model_name': artifacts.get('metadata', {}).get('strategy', model_key),
    }


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def train_all_models(verbose=True):
    """
    Execute all production model training scripts and collect results.

    Args:
        verbose: If True, stream each script's stdout to console.

    Returns:
        tuple: (results_summary dict, comparison DataFrame)
    """
    print("\n" + "=" * 80)
    print("PRODUCTION ML TRAINING ORCHESTRATOR".center(80))
    print("=" * 80)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models    : {len(MODEL_SCRIPTS)}")
    print(f"  Python    : {sys.executable}")
    print("=" * 80 + "\n")

    results = {}
    errors = {}

    for i, entry in enumerate(MODEL_SCRIPTS, 1):
        key = entry['key']
        name = entry['name']
        script = entry['script']

        print(f"\n[{i}/{len(MODEL_SCRIPTS)}] Training {name}")
        print("-" * 80)

        if not os.path.exists(script):
            msg = f"Script not found: {script}"
            print(f"  ✗ {msg}")
            errors[key] = msg
            continue

        # Run training script as isolated subprocess
        rc, stdout, stderr = run_model_script(script)

        if verbose and stdout:
            # Indent subprocess output for readability
            for line in stdout.strip().split('\n'):
                print(f"  | {line}")

        if rc != 0:
            msg = stderr.strip().split('\n')[-1] if stderr.strip() else f"Exit code {rc}"
            print(f"  ✗ Training FAILED: {msg}")
            if verbose and stderr:
                for line in stderr.strip().split('\n')[-5:]:
                    print(f"  ! {line}")
            errors[key] = msg
            continue

        # Load metrics from saved .pkl
        metrics = load_model_metrics(entry['pkl'])
        if metrics:
            results[key] = {'name': name, **metrics}
            print(f"  ✓ Complete — F1: {metrics['f1']:.4f}  "
                  f"Prec: {metrics['precision']:.4f}  "
                  f"Rec: {metrics['recall']:.4f}  "
                  f"AUC: {metrics['roc_auc']:.4f}  "
                  f"Thr: {metrics['threshold']:.3f}")
        else:
            msg = f"Could not load {entry['pkl']}"
            print(f"  ✗ {msg}")
            errors[key] = msg

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON".center(80))
    print("=" * 80 + "\n")

    if results:
        rows = []
        for entry in MODEL_SCRIPTS:
            key = entry['key']
            if key in results:
                r = results[key]
                rows.append({
                    'Model': r['name'],
                    'Strategy': r.get('strategy', 'N/A'),
                    'Threshold': r['threshold'],
                    'Accuracy': r['accuracy'],
                    'Precision': r['precision'],
                    'Recall': r['recall'],
                    'F1-Score': r['f1'],
                    'ROC-AUC': r['roc_auc'],
                })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

        # Winners
        print("\n" + "-" * 80)
        best_f1 = df['F1-Score'].idxmax()
        best_acc = df['Accuracy'].idxmax()
        best_prec = df['Precision'].idxmax()
        best_rec = df['Recall'].idxmax()
        best_auc = df['ROC-AUC'].idxmax()

        print(f"  Best F1-Score  : {df.loc[best_f1, 'Model']}  ({df.loc[best_f1, 'F1-Score']:.4f})")
        print(f"  Best Accuracy  : {df.loc[best_acc, 'Model']}  ({df.loc[best_acc, 'Accuracy']:.4f})")
        print(f"  Best Precision : {df.loc[best_prec, 'Model']}  ({df.loc[best_prec, 'Precision']:.4f})")
        print(f"  Best Recall    : {df.loc[best_rec, 'Model']}  ({df.loc[best_rec, 'Recall']:.4f})")
        print(f"  Best ROC-AUC   : {df.loc[best_auc, 'Model']}  ({df.loc[best_auc, 'ROC-AUC']:.4f})")
    else:
        df = pd.DataFrame()
        print("  No models trained successfully.")

    if errors:
        print("\n  Errors:")
        for k, v in errors.items():
            print(f"    {k}: {v}")

    # ============================================================
    # SAVE TRAINING REPORT
    # ============================================================
    report = {
        'training_timestamp': datetime.now().isoformat(),
        'pipeline': 'production_orchestrator',
        'notes': [
            'Each model uses its own production-grade training script',
            'BalancedBagging / SMOTE / ensemble strategies per model',
            'Validation-based F1 threshold optimization (no test leakage)',
            'Grid search over model-specific hyperparameters',
        ],
        'models': results,
        'errors': errors,
        'comparison': df.to_dict('records') if not df.empty else [],
    }

    report_path = 'model/training_report.json'
    os.makedirs('model', exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=_convert)

    print(f"\n  Training report saved: {report_path}")

    print("\n" + "=" * 80)
    print("ALL TRAINING COMPLETE".center(80))
    print("=" * 80 + "\n")

    return results, df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results, comparison_df = train_all_models(verbose=True)

    # Quick post-training verification
    print("[VERIFICATION] Loading each saved model...\n")
    for entry in MODEL_SCRIPTS:
        try:
            arts = load_trained_model(entry['key'])
            n_feat = len(arts.get('feature_names', []))
            thr = arts.get('threshold', 'N/A')
            strategy = arts.get('metadata', {}).get('strategy', 'N/A')
            print(f"  ✓ {entry['name']:25s}  features={n_feat}  threshold={thr}  strategy={strategy}")
        except Exception as e:
            print(f"  ✗ {entry['name']:25s}  {e}")

    print("\n✅ All models trained and saved!")
    print("   Run 'streamlit run app.py' to launch the inference dashboard.")
