import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ============================================================
# FEATURE ENGINEERING (Production-Grade)
# ============================================================

def engineer_features(df):
    """
    Create domain-specific cardiovascular risk features.
    Based on clinical research and Framingham Heart Study literature.

    These interaction and non-linear features allow linear models (like
    Logistic Regression) to capture complex risk patterns that raw
    features alone cannot represent.

    Args:
        df: DataFrame with standard Framingham columns

    Returns:
        DataFrame with original + engineered features
    """
    df = df.copy()

    # 1. Pulse Pressure — proven arterial stiffness / CVD risk indicator
    df['pulse_pressure'] = df['sysBP'] - df['diaBP']

    # 2. Mean Arterial Pressure — overall blood pressure metric
    df['map_bp'] = df['diaBP'] + (df['sysBP'] - df['diaBP']) / 3

    # 3. Age x Systolic BP — age amplifies hypertension risk
    df['age_sysBP'] = df['age'] * df['sysBP']

    # 4. Metabolic risk — BMI x glucose (metabolic syndrome proxy)
    df['metabolic_risk'] = df['BMI'] * df['glucose']

    # 5. Age squared — non-linear aging effect on cardiac risk
    df['age_squared'] = df['age'] ** 2

    # 6. Clinical threshold indicators (evidence-based cutoffs)
    df['hypertension_flag'] = (df['sysBP'] >= 140).astype(int)
    df['high_chol_flag'] = (df['totChol'] >= 240).astype(int)
    df['diabetes_flag'] = (df['glucose'] >= 126).astype(int)

    # 7. Smoking intensity and cumulative damage
    df['smoking_pack_years'] = df['currentSmoker'] * df['cigsPerDay']
    df['age_smoking'] = df['age'] * df['cigsPerDay'] / 100

    # 8. Additional interactions
    df['bmi_bp'] = df['BMI'] * df['sysBP']
    df['male_age'] = df['male'] * df['age']

    # 9. Targeted high-value interactions (top correlated features)
    df['age_diaBP'] = df['age'] * df['diaBP']
    df['age_glucose'] = df['age'] * df['glucose']
    df['sysBP_chol'] = df['sysBP'] * df['totChol']
    df['male_sysBP'] = df['male'] * df['sysBP']
    df['age_prevalentHyp'] = df['age'] * df['prevalentHyp']
    df['age_BMI'] = df['age'] * df['BMI']

    # 10. Risk Factor Count — total modifiable risk factors present
    #     Clinically proven: cumulative risk burden is the strongest predictor
    df['risk_factor_count'] = (
        (df['sysBP'] >= 140).astype(int) +           # Hypertension
        (df['totChol'] >= 240).astype(int) +          # High cholesterol
        (df['glucose'] >= 126).astype(int) +          # Diabetes/pre-diabetes
        (df['BMI'] >= 30).astype(int) +               # Obesity
        (df['currentSmoker'] == 1).astype(int) +      # Current smoker
        (df['BPMeds'] == 1).astype(int) +             # On BP medication
        (df['prevalentHyp'] == 1).astype(int) +       # Prevalent hypertension
        (df['prevalentStroke'] == 1).astype(int)       # Prevalent stroke
    )

    # 11. Log transforms — normalize right-skewed distributions
    #     Helps LR by reducing outlier influence and improving linearity
    df['log_glucose'] = np.log1p(df['glucose'])
    df['log_totChol'] = np.log1p(df['totChol'])
    df['log_cigsPerDay'] = np.log1p(df['cigsPerDay'])

    # 12. Age-risk interaction — age amplifies cumulative risk
    df['age_risk_interaction'] = df['age'] * df['risk_factor_count']

    # 13. BMI categories (clinical cutoffs)
    df['bmi_overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
    df['bmi_obese'] = (df['BMI'] >= 30).astype(int)

    return df


# ============================================================
# PART 1: DATA SPLITTING
# ============================================================

def split_dataset(source_filepath='data/framingham_heart_study.csv', 
                  train_output='data/train_framingham.csv',
                  test_output='data/test_framingham.csv',
                  test_size=0.2,
                  random_state=42,
                  stratify_col='TenYearCHD'):
    """
    Split original dataset into train and test sets.
    Uses stratification to maintain class distribution.
    
    Args:
        source_filepath: Path to original dataset
        train_output: Path to save training data
        test_output: Path to save test data
        test_size: Proportion of data for testing (default 0.2 = 20%)
        random_state: Random seed for reproducibility
        stratify_col: Column to stratify by (for balanced splitting)
    
    Returns:
        tuple: (train_df, test_df) with summary statistics
    """
    if not os.path.exists(source_filepath):
        raise FileNotFoundError(f"Source file not found: {source_filepath}")
    
    # Load original data
    df = pd.read_csv(source_filepath)
    print(f"Original dataset: {len(df)} rows")
    
    # Split with stratification
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Save to files
    df_train.to_csv(train_output, index=False)
    df_test.to_csv(test_output, index=False)
    
    print(f"Training set: {len(df_train)} rows (80%)")
    print(f"Test set: {len(df_test)} rows (20%)")
    
    # Show class distribution if target column exists
    if stratify_col and stratify_col in df.columns:
        print(f"\nClass Distribution [{stratify_col}]:")
        print(f"  Original - 0: {(df[stratify_col] == 0).sum()}, 1: {(df[stratify_col] == 1).sum()}")
        print(f"  Train    - 0: {(df_train[stratify_col] == 0).sum()}, 1: {(df_train[stratify_col] == 1).sum()}")
        print(f"  Test     - 0: {(df_test[stratify_col] == 0).sum()}, 1: {(df_test[stratify_col] == 1).sum()}")
    
    return df_train, df_test


# ============================================================
# PART 2: DATA PREPARATION (For Training)
# ============================================================

def prepare_training_data(filepath='data/train_framingham.csv',
                         target_col='TenYearCHD',
                         apply_oversampling=True,
                         validation_size=0.15):
    """
    Production-grade training data pipeline:
    1. Load data, impute missing values (preserve all samples)
    2. Engineer domain-specific features
    3. Split into train/validation (BEFORE oversampling)
    4. Fit scaler on training portion
    5. Apply SMOTE on training portion only (proper methodology)

    Args:
        filepath: Path to training dataset
        target_col: Name of target column
        apply_oversampling: Whether to apply SMOTE oversampling
        validation_size: Fraction for validation set (0 = no validation split)

    Returns:
        tuple: (X_scaled, y, scaler, feature_names, stats_dict)
               stats_dict includes imputation_values and optional validation data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    # 1. Load Dataset
    df = pd.read_csv(filepath)
    print(f"Loaded training data: {len(df)} rows")

    # 2. Impute Missing Values (median strategy — preserves ALL samples)
    imputation_values = {}
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].isnull().any():
            imputation_values[col] = float(df[col].median())
    initial_missing = df.isnull().sum().sum()
    df_clean = df.fillna(imputation_values)
    df_clean = df_clean.reset_index(drop=True)
    print(f"Imputed {initial_missing} missing values across {len(imputation_values)} columns")
    print(f"All {len(df_clean)} samples preserved")

    # 3. Feature Engineering (create domain-specific risk features)
    df_clean = engineer_features(df_clean)

    # 4. Define Features and Target
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    feature_names = X.columns.tolist()
    n_original = len([c for c in df.columns if c != target_col])
    n_engineered = len(feature_names) - n_original
    print(f"Features: {len(feature_names)} ({n_original} original + {n_engineered} engineered)")

    # 5. Train/Validation split (BEFORE oversampling — critical for unbiased evaluation)
    X_val_scaled = None
    y_val = None
    if validation_size > 0:
        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=42, stratify=y
        )
        print(f"Train/Val split: {len(X_train_raw)} train + {len(X_val_raw)} validation")
    else:
        X_train_raw = X
        y_train = y
        X_val_raw = None

    # 6. Feature Scaling (fit on training split only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    if X_val_raw is not None:
        X_val_scaled = scaler.transform(X_val_raw)
        y_val = y_val.reset_index(drop=True)

    # 7. SMOTE Oversampling (on training portion only — production best practice)
    if apply_oversampling:
        pre_smote = len(X_train_scaled)
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        print(f"SMOTE: {pre_smote} -> {len(X_train_scaled)} samples (balanced)")
        print(f"  Class 0: {int(np.sum(y_train == 0))}, Class 1: {int(np.sum(y_train == 1))}")
    else:
        print(f"Class distribution (no oversampling):")
        print(f"  Class 0: {int(np.sum(y_train == 0))}, Class 1: {int(np.sum(y_train == 1))}")

    stats = {
        'n_samples': len(X_train_scaled),
        'n_features': len(feature_names),
        'class_distribution': {0: int(np.sum(y_train == 0)), 1: int(np.sum(y_train == 1))},
        'imputation_values': imputation_values,
        'X_val': X_val_scaled,
        'y_val': np.array(y_val) if y_val is not None else None
    }

    return X_train_scaled, y_train, scaler, feature_names, stats


# ============================================================
# PART 3: DATA PREPARATION (For Testing/Inference)
# ============================================================

def prepare_test_data(filepath='data/test_framingham.csv',
                      scaler=None,
                      feature_names=None,
                      target_col='TenYearCHD',
                      imputation_values=None):
    """
    Prepare test/inference data using pre-fitted scaler and imputation values.
    Used during model evaluation or inference.

    Production critical:
    - Always use the scaler fitted on training data
    - Always use imputation values from training data
    - Apply same feature engineering as training

    Args:
        filepath: Path to test/inference dataset
        scaler: Pre-fitted StandardScaler (from training)
        feature_names: Feature order (from training)
        target_col: Name of target column (None for inference)
        imputation_values: Dict of {column: median_value} from training

    Returns:
        tuple: (X_scaled, y_true, stats_dict) where y_true is None if target not in data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}")

    if scaler is None:
        raise ValueError("Scaler must be provided (use scaler from training data)")

    if feature_names is None:
        raise ValueError("Feature names must be provided (use from training data)")

    # 1. Load Dataset
    df = pd.read_csv(filepath)
    print(f"Loaded test data: {len(df)} rows")

    # 2. Handle Missing Values (impute using training medians, or drop as fallback)
    if imputation_values:
        initial_missing = df.isnull().sum().sum()
        df_clean = df.fillna(imputation_values)
        df_clean = df_clean.reset_index(drop=True)
        if initial_missing > 0:
            print(f"Imputed {initial_missing} missing values (using training medians)")
        print(f"All {len(df_clean)} test samples preserved")
    else:
        initial_rows = len(df)
        df_clean = df.dropna()
        df_clean = df_clean.reset_index(drop=True)
        rows_dropped = initial_rows - len(df_clean)
        if rows_dropped > 0:
            print(f"Dropped {rows_dropped} rows with missing values")

    # 3. Feature Engineering (same transformations as training)
    df_clean = engineer_features(df_clean)

    # 4. Extract Target if Present
    y_true = None
    if target_col and target_col in df_clean.columns:
        X = df_clean.drop(columns=[target_col])
        y_true = df_clean[target_col].reset_index(drop=True)
        print(f"Target found. Class distribution: 0: {(y_true == 0).sum()}, 1: {(y_true == 1).sum()}")
    else:
        X = df_clean
        print("No target column found (inference mode)")

    # 5. Ensure Correct Features
    X = X[feature_names]  # Reorder to match training features

    # 6. Scale using Training Scaler (DO NOT FIT!)
    X_scaled = scaler.transform(X)

    stats = {
        'n_samples': len(X),
        'n_features': len(feature_names),
        'has_target': y_true is not None,
        'class_distribution': {0: int((y_true == 0).sum()), 1: int((y_true == 1).sum())} if y_true is not None else None
    }

    return X_scaled, y_true, stats


# ============================================================
# PART 4: LEGACY FUNCTION (Backward Compatibility)
# ============================================================

def prepare_data(filepath='data/train_framingham.csv',
                test_filepath='data/test_framingham.csv',
                target_col='TenYearCHD',
                apply_oversampling=True):
    """
    Complete data preparation pipeline (backward compatible).
    Loads train and test data, processes train (with oversampling),
    and applies pre-fitted scaler to test.
    
    Args:
        filepath: Path to training dataset
        test_filepath: Path to test dataset
        target_col: Name of target column
        apply_oversampling: Whether to apply oversampling to training data
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # Prepare training data
    X_train_scaled, y_train, scaler, feature_names, train_stats = prepare_training_data(
        filepath=filepath,
        target_col=target_col,
        apply_oversampling=apply_oversampling,
        validation_size=0  # No validation for legacy function
    )
    
    # Prepare test data (using training imputation values)
    X_test_scaled, y_test, _ = prepare_test_data(
        filepath=test_filepath,
        scaler=scaler,
        feature_names=feature_names,
        target_col=target_col,
        imputation_values=train_stats.get('imputation_values')
    )
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    print("=== Data Preparation Module ===\n")
    
    # Step 1: Split original data (one-time operation)
    print("Step 1: Splitting original dataset...")
    try:
        split_dataset()
        print()
    except FileNotFoundError:
        print("Original dataset not found. Using existing train/test files.\n")
    
    # Step 2: Prepare data for training
    print("Step 2: Preparing training data (imputation + feature engineering + SMOTE)...")
    X_train, y_train, scaler, features, train_stats = prepare_training_data()
    print(f"Training data shape: {X_train.shape}\n")
    
    # Step 3: Prepare test data
    print("Step 3: Preparing test data...")
    X_test, y_test, test_stats = prepare_test_data(
        scaler=scaler, feature_names=features,
        imputation_values=train_stats.get('imputation_values')
    )
    print(f"Test data shape: {X_test.shape}\n")
    
    print("Data preparation complete!")
    print(f"  Features: {len(features)} ({features})")