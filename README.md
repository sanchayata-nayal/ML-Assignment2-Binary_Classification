# Comparative Analysis of Classification Algorithms for Coronary Heart Disease Prediction

## 1. Executive Summary

The primary objective of this research is to develop and evaluate a machine learning framework capable of predicting the ten-year risk of Coronary Heart Disease (CHD). Utilizing data from the Framingham Heart Study, this project implements a binary classification pipeline. Six distinct algorithms were trained and rigorously evaluated to identify the most efficacious model for medical risk assessment. The final deliverable includes a deployed Streamlit web application for real-time inference.

## 2. Dataset Methodology

**Source:** Framingham Heart Study (Public Health Dataset)
**Problem Type:** Binary Classification
**Target Variable:** `TenYearCHD` (0 = No Risk; 1 = Risk present)

**Data Characteristics:**
The dataset comprises 15 independent variables representing patient health profiles.

* **Demographic Factors:** Age, Sex, Education Level.

* **Behavioral Metrics:** Smoking Status, Daily Cigarette Consumption.

* **Medical History:** Blood Pressure Medication, Prevalent Stroke, Hypertension, Diabetes.

* **Clinical Measurements:** Total Cholesterol, Systolic/Diastolic BP, BMI, Heart Rate, Glucose.

**Preprocessing Protocols:**

* **Data Cleaning:** Instances containing missing values were removed, reducing the sample size from 4,240 to approximately 3,658 to ensure data integrity.

* **Feature Scaling:** A `StandardScaler` was applied to normalize the range of independent variables, mitigating bias in distance-based algorithms (e.g., kNN).

* **Data Splitting:** The dataset was partitioned into a training set (80%) and a testing set (20%) using stratified sampling to maintain class distribution.

## 3. Experimental Results

The following table presents the performance metrics derived from the test set evaluation.

| Model | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC | 
 | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Logistic Regression** | **0.8620** | **0.7257** | 0.7273 | 0.1441 | 0.2406 | **0.2825** | 
| **Decision Tree** | 0.8388 | 0.6600 | 0.3704 | 0.0901 | 0.1449 | 0.1193 | 
| **k-Nearest Neighbors** | 0.8402 | 0.6460 | 0.3750 | 0.0811 | 0.1333 | 0.1147 | 
| **Naive Bayes** | 0.8210 | 0.6903 | 0.3571 | **0.2252** | **0.2762** | 0.1863 | 
| **Random Forest** | 0.8566 | 0.7140 | **0.8000** | 0.0721 | 0.1322 | 0.2127 | 
| **XGBoost** | 0.8279 | 0.6528 | 0.3469 | 0.1532 | 0.2125 | 0.1458 | 

## 4. Analysis and Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the **highest Accuracy (86.2%)** and **AUC (0.7257)**. It proved to be the most robust baseline model for this dataset, effectively handling the linear relationships between risk factors and heart disease. |
| **Decision Tree** | Exhibited lower AUC (0.66) compared to the linear baseline. Despite depth limiting to prevent overfitting, it struggled to capture the subtle probabilistic nature of heart disease risk compared to Logistic Regression. |
| **KNN** | Performance was average with an AUC of 0.64. The algorithm likely struggled because the "distance" separation between healthy and at-risk patients in the 15-dimensional feature space is not distinct enough without more advanced feature engineering. |
| **Naive Bayes** | While having lower overall accuracy, it achieved the **highest Recall (22.5%)** and **Best F1 Score**. In a medical context, this sensitivity makes it valuable for screening as it misses fewer positive cases than other models. |
| **Random Forest** | Achieved the **highest Precision (80.0%)**. This model is highly conservative; when it predicts "Risk", it is usually correct. However, this conservatism resulted in a very low Recall (7.2%), meaning it missed many actual cases. |
| **XGBoost** | Surprisingly performed worse than the simple Logistic Regression on this specific dataset (AUC 0.65). This suggests that the dataset might not have complex non-linear patterns that boosting algorithms typically exploit, or it requires extensive hyperparameter tuning. |

## 5. Project Architecture

The repository is organized to facilitate reproducibility and deployment:

├── app.py                 # Streamlit application entry point
├── data_prep.py           # Data loading and preprocessing logic
├── framingham.csv         # Raw dataset
├── model/                 # Serialized models and training scripts
│   ├── decision_tree.py
│   ├── knn.py
│   ├── logistic_regression.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   └── *.pkl              # Persisted model artifacts
├── README.md              # Project documentation
└── requirements.txt       # Dependency configuration

## 6. Technical Implementation

### 6.1 Prerequisites

* Python 3.8 or higher

* PIP Package Manager

### 6.2 Installation Instructions

1. **Clone the Repository:**
git clone <repository_url>

2. **Environment Configuration:**
It is recommended to utilize a virtual environment to manage dependencies.
pip install -r requirements.txt

### 6.3 Application Deployment
The interactive dashboard is built using Streamlit. To launch the application locally:
streamlit run app.py
