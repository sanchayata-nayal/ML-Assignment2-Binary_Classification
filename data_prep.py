import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(filepath='framingham_heart_study.csv'):
    """
    Loads data, drops nulls, splits into train/test, and performs 
    Random Oversampling ONLY on the training set to prevent data leakage.
    Returns: X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
    """
    # 1. Load Dataset
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {filepath}. Check the path!")

    # 2. Handle Missing Values
    df_clean = df.dropna()
    
    # 3. Define Features (X) and Target (y)
    target_col = 'TenYearCHD'
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    feature_names = X.columns.tolist()

    # 4. Train-Test Split (80% Train, 20% Test)
    # STRATIFY is crucial here to ensure both sets have some positive cases initially
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 5. OVERSAMPLING (Applied ONLY to Training Data) ---
    # Recombine X_train and y_train temporarily
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Separate classes
    majority = train_data[train_data[target_col] == 0]
    minority = train_data[train_data[target_col] == 1]
    
    # Duplicate Minority Class (4x as initial)
    minority_upsampled = pd.concat([minority] * 2, axis=0)
    
    # Combine back
    train_balanced = pd.concat([majority, minority_upsampled])
    
    # Shuffle the training data
    train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate back into X and y
    X_train_balanced = train_balanced.drop(columns=[target_col])
    y_train_balanced = train_balanced[target_col]
    
    print(f"Original Training Count: {len(X_train)}")
    print(f"Balanced Training Count: {len(X_train_balanced)} (Positives increased 4x)")

    # 6. Feature Scaling
    # Fit scaler on the BALANCED training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    
    # Transform test data using the SAME scaler (do not refit on test)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler, feature_names

if __name__ == "__main__":
    # Test run
    X_tr, X_te, y_tr, y_te, _, _ = prepare_data()
    print(f"Final Training shape: {X_tr.shape}")