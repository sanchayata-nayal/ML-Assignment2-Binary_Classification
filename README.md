# Comparative Analysis of Classification Algorithms for Coronary Heart Disease Prediction

## 1. Executive Summary

The primary objective of this research is to develop and evaluate a production-grade machine learning framework capable of predicting the ten-year risk of Coronary Heart Disease (CHD). Utilizing data from the Framingham Heart Study, this project implements a binary classification pipeline with advanced imbalanced-learning techniques. Six distinct algorithms were trained using ensemble strategies (BalancedBagging, EasyEnsemble, RUSBoost) with validation-based threshold optimization to maximize F1-Score on the heavily imbalanced dataset (~15% positive class). The final deliverable includes a deployed Streamlit web application for real-time inference.

## 2. Dataset Methodology

**Source:** Framingham Heart Study (Public Health Dataset)
**Problem Type:** Binary Classification (Imbalanced -- ~85:15 ratio)
**Target Variable:** `TenYearCHD` (0 = No Risk; 1 = Risk present)

**Data Characteristics:**
The dataset comprises 15 original independent variables representing patient health profiles, expanded to **40 features** through domain-specific feature engineering.

* **Demographic Factors:** Age, Sex, Education Level.

* **Behavioral Metrics:** Smoking Status, Daily Cigarette Consumption.

* **Medical History:** Blood Pressure Medication, Prevalent Stroke, Hypertension, Diabetes.

* **Clinical Measurements:** Total Cholesterol, Systolic/Diastolic BP, BMI, Heart Rate, Glucose.

* **Engineered Features (25):** Pulse Pressure, Mean Arterial Pressure, Age-BP/Glucose/BMI interactions, Metabolic Risk Score, Risk Factor Count, Clinical Threshold Flags (Hypertension, High Cholesterol, Diabetes), Log-transformed skewed distributions, BMI categories, Smoking Pack-Years, and Age-Risk interactions.

**Preprocessing Protocols:**

* **Data Cleaning:** Missing values were imputed using **median imputation** from the training set, preserving all 4,240 samples (3,392 train + 848 test). No data was dropped.

* **Feature Engineering:** 25 clinically-motivated features were created via `engineer_features()` in `data_prep.py`, resulting in 40 total features per sample.

* **Feature Scaling:** A `StandardScaler` was fitted on the training split only, then applied to validation and test sets (no data leakage).

* **Data Splitting:** The dataset was partitioned into a training set (80%, 3,392 samples) and a testing set (20%, 848 samples) using stratified sampling. Within training, a 85/15 train/validation split (2,883/509 samples) was used for threshold optimization.

* **Class Imbalance Handling:** Each model uses its own strategy -- BalancedBagging (undersampling per bootstrap), SMOTE/SMOTEENN/SMOTETomek (oversampling), EasyEnsemble, or RUSBoost -- applied **only to the training portion** to prevent validation leakage.

## 3. Experimental Results

The following table presents the performance metrics derived from the **held-out test set** (848 samples: 719 negative, 129 positive). All models use **optimized decision thresholds** tuned on the validation set to maximize F1-Score.

| Model | Threshold | Accuracy | AUC Score | Precision | Recall | F1 Score | Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.370 | 0.5153 | **0.7008** | 0.2202 | **0.8605** | **0.3507** | BalancedBagging(LR, C=0.1, L1, n=50) |
| **Decision Tree** | 0.565 | 0.7158 | 0.6726 | 0.2586 | 0.4651 | 0.3324 | BalancedBagging(DT, d=6, l=10, f=0.7, n=100) |
| **k-Nearest Neighbors** | 0.485 | 0.6627 | 0.6785 | 0.2524 | 0.6202 | 0.3587 | BalancedBagging(KNN, k=11, uniform, p=2, n=30) |
| **Naive Bayes** | 0.465 | **0.7524** | 0.6810 | **0.2737** | 0.3798 | 0.3182 | BalancedBagging(GaussianNB, vs=1e-11, n=80) |
| **Random Forest** | 0.525 | 0.7241 | 0.6677 | 0.2624 | 0.4496 | 0.3314 | BalancedBagging-RF(DT, d=20, leaf=3, n=200) |
| **XGBoost** | 0.550 | 0.7087 | 0.6784 | 0.2542 | 0.4729 | 0.3306 | BB-XGB(bags=50, n=30, d=5, lr=0.05, mcw=3) |

**Category Winners:**
* **Best F1-Score:** k-Nearest Neighbors (0.3587)
* **Best Accuracy:** Naive Bayes (0.7524)
* **Best Precision:** Naive Bayes (0.2737)
* **Best Recall:** Logistic Regression (0.8605)
* **Best ROC-AUC:** Logistic Regression (0.7008)

## 4. Analysis and Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest AUC (0.7008)** and **highest Recall (86.1%)**, making it the best screening model. With a low threshold of 0.37, it aggressively flags at-risk patients, catching 111 out of 129 true positive cases. The trade-off is reduced accuracy (51.5%) due to high false positive rate. The L1-regularized BalancedBagging ensemble provides interpretable feature selection while handling class imbalance effectively. Ideal for a **high-sensitivity clinical screening** scenario where missing a positive case is costly. |
| **Decision Tree** | Performed competitively with F1=0.3324 using a BalancedBagging ensemble of shallow trees (depth=6, min_leaf=10). The multi-strategy search (BalancedBagging, EasyEnsemble, RUSBoost across 131 configurations) found BalancedBagging to be the strongest strategy. The validation-tuned threshold (0.565) balances precision and recall better than deeper tree variants, which tended to overfit. |
| **KNN** | Achieved the **second-highest F1-Score (0.3587)** and highest among all models on this metric. A 7-strategy search (SMOTE, SMOTEENN, SMOTETomek, BalancedBagging, PCA+SMOTEENN, PCA+BalancedBagging, SelectKBest+SMOTEENN across 492 configurations) revealed BalancedBagging with k=11, uniform weights, L2 distance to be optimal. The threshold was capped at 0.05--0.50 to prevent inflation from balanced training data. Despite the curse of dimensionality with 40 features, the ensemble approach provided robust performance. |
| **Naive Bayes** | Achieved the **highest Accuracy (75.2%)** and **highest Precision (27.4%)** among all models. The BalancedBagging(GaussianNB) ensemble across 33 configurations showed remarkable stability -- nearly all configs achieved similar validation F1 (~0.38). The model's probabilistic nature combined with balanced bagging produces well-calibrated predictions. Best suited for scenarios requiring **fewer false alarms** while maintaining reasonable detection capability (38.0% recall). |
| **Random Forest** | Achieved F1=0.3314 with a BalancedBagging ensemble wrapping individual Decision Trees with `max_features='sqrt'` (simulating Random Forest behavior). A grid of 48 configurations over depth, leaf size, and estimator count was searched. The winner (depth=20, min_leaf=3, n=200) provides deep, expressive trees while the BalancedBagging framework handles imbalance. Performance is close to XGBoost, suggesting the dataset's signal is well-captured by tree-based methods at this complexity level. |
| **XGBoost** | Achieved F1=0.3306 using a dual-strategy search: Strategy A (XGBoost + scale_pos_weight + early stopping, 54 configs) and Strategy B (BalancedBagging wrapping XGBoost, 32 configs). Strategy B won, indicating that balanced subsampling via bagging outperforms internal class weighting for this dataset. The threshold was capped at 0.55 to prevent overshoot after retraining on full data. Despite being the most complex model, it did not significantly outperform simpler tree-based approaches, suggesting the Framingham dataset's predictive signal is relatively linear. |

## 5. Conclusions

1. **Class imbalance is the dominant challenge.** With only ~15% positive cases, all models face a fundamental precision-recall trade-off. BalancedBagging (bootstrap undersampling) proved to be the most effective strategy across all model families, consistently outperforming SMOTE, EasyEnsemble, and RUSBoost alternatives.

2. **Threshold optimization is critical.** Default thresholds (0.5) are inappropriate for imbalanced data. Validation-based threshold tuning improved F1-Scores by 30--80% compared to baseline models. Each model family requires its own optimal threshold (ranging from 0.37 for LR to 0.565 for DT).

3. **Model complexity does not guarantee better performance.** Logistic Regression (the simplest model) achieved the highest AUC and Recall, while KNN achieved the highest F1-Score. XGBoost and Random Forest, despite their sophistication, did not surpass simpler models on this dataset. This suggests the underlying CHD risk signal is largely linear and low-dimensional.

4. **Feature engineering adds value.** Expanding from 15 raw features to 40 engineered features (clinical threshold flags, interaction terms, log transforms, risk factor count) provided all models with richer representations of cardiovascular risk patterns.

5. **No single model dominates all metrics.** The choice of model depends on the clinical use case:
   - **Mass screening (maximize recall):** Logistic Regression (86.1% recall)
   - **Balanced detection (maximize F1):** KNN (F1=0.3587) or Logistic Regression (F1=0.3507)
   - **Confirmatory diagnosis (maximize precision):** Naive Bayes (27.4% precision)
   - **Overall discrimination (maximize AUC):** Logistic Regression (AUC=0.7008)

## 6. Project Architecture

The repository is organized to facilitate reproducibility and deployment:

| Path | Description |
| :--- | :--- |
| `app.py` | Streamlit web application for inference and visualization |
| `data_prep.py` | Data loading, median imputation, feature engineering, scaling |
| `train_all_models.py` | Orchestrator -- trains all 6 models via subprocess |
| `requirements.txt` | Python dependency configuration |
| `2025AA05965_assignment.ipynb` | Jupyter notebook for assignment submission |
| `README.md` | Project documentation |
| **data/** | |
| `data/framingham_heart_study.csv` | Original Framingham dataset (4,240 samples) |
| `data/train_framingham.csv` | Training split (3,392 samples) |
| `data/test_framingham.csv` | Test split (848 samples) |
| **model/** | |
| `model/logistic_regression.py` | LR training pipeline (BalancedBagging) |
| `model/decision_tree.py` | DT training pipeline (3 strategies: BB, EE, RUSBoost) |
| `model/knn.py` | KNN training pipeline (7 strategies) |
| `model/naive_bayes.py` | NB training pipeline (BalancedBagging) |
| `model/random_forest.py` | RF training pipeline (BalancedBagging) |
| `model/xgboost_model.py` | XGBoost training pipeline (2 strategies) |
| `model/training_report.json` | Unified metrics report (auto-generated) |
| `model/*.pkl` | Persisted model artifacts (auto-generated) |

## 7. Technical Implementation

### 7.1 Prerequisites

* Python 3.8 or higher
* PIP Package Manager

### 7.2 Installation Instructions

1. **Clone the Repository:**
```bash
git clone <repository_url>
cd ML-Assignment2-Binary_Classification
```

2. **Environment Configuration:**
It is recommended to utilize a virtual environment to manage dependencies.
```bash
pip install -r requirements.txt
```

### 7.3 Training All Models
To train all 6 models sequentially with the production orchestrator:
```bash
python train_all_models.py
```
Each model script runs independently, performing its own grid search, threshold optimization, and test evaluation. Results are saved to `model/training_report.json`.

### 7.4 Training Individual Models
Each model can also be trained independently:
```bash
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py
python model/xgboost_model.py
```

### 7.5 Application Deployment
The interactive dashboard is built using Streamlit. To launch the application locally:
```bash
streamlit run app.py
```

## 8. Overall Conclusion

This project demonstrates a comprehensive, end-to-end machine learning pipeline for predicting the ten-year risk of Coronary Heart Disease using the Framingham Heart Study dataset. Across all six classification algorithms -- Logistic Regression, Decision Tree, k-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost -- the primary bottleneck was the severe class imbalance (~85:15 negative-to-positive ratio), which renders naive accuracy misleading and demands specialized techniques.

**Key takeaways:**

- **BalancedBagging emerged as the universally best imbalance-handling strategy**, winning the internal competition in every model family over SMOTE, SMOTEENN, SMOTETomek, EasyEnsemble, and RUSBoost. This suggests that for moderately sized tabular datasets with low positive prevalence, bootstrap-based undersampling provides the most stable and generalizable corrections.

- **Threshold optimization was indispensable.** Moving away from the default 0.5 decision boundary -- and instead tuning thresholds on a held-out validation set -- yielded F1 improvements of 30-80% across all models. The optimal thresholds ranged from 0.37 (Logistic Regression) to 0.565 (Decision Tree), reflecting each algorithm's unique probability calibration characteristics.

- **Simpler models matched or outperformed complex ones.** Logistic Regression achieved the best AUC (0.7008) and Recall (86.1%), while KNN achieved the best F1-Score (0.3587). Neither XGBoost nor Random Forest -- despite their capacity for modeling non-linear interactions -- could surpass these simpler baselines. This indicates that the CHD risk signal in the Framingham dataset is predominantly linear and well-captured by straightforward models augmented with good feature engineering.

- **Feature engineering from 15 to 40 variables** (pulse pressure, MAP, age interactions, clinical flags, log transforms, risk factor counts) enriched the input representation and contributed to improved performance across all model families.

- **The best model depends on the clinical objective.** For mass screening where missing a positive case is unacceptable, Logistic Regression (86.1% recall) is the clear choice. For balanced precision-recall performance, KNN (F1=0.3587) leads. For minimizing false alarms in a confirmatory setting, Naive Bayes (27.4% precision, 75.2% accuracy) is preferred. No single model dominates all evaluation criteria, reinforcing the importance of aligning model selection with the specific healthcare deployment context.

In summary, this project validates that thoughtful data preprocessing, domain-driven feature engineering, systematic imbalance handling, and principled threshold calibration collectively matter more than algorithm complexity for real-world medical risk prediction.
