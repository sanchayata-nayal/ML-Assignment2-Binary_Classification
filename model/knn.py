import sys
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)


# --- 1. Import Data prep ---
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

# --- 3. Train kNN ---
print("Training k-Nearest Neighbors (kNN)...")
# n_neighbors=5 is a standard starting point
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# --- 4. Evaluate ---
y_pred = knn_model.predict(X_test)
y_prob = knn_model.predict_proba(X_test)[:, 1]

print("\n--- kNN Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC:       {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")

# --- 5. Save Model ---
if 'model' in os.listdir(parent_dir):
    save_dir = os.path.join(parent_dir, 'model')
else:
    save_dir = os.path.dirname(os.path.abspath(__file__))

model_data = {
    'model': knn_model,
    'scaler': scaler,
    'feature_names': feature_names
}

output_path = os.path.join(save_dir, 'knn_model.pkl')
joblib.dump(model_data, output_path)
print(f"\nModel saved to {output_path}")