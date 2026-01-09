import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(filepath='framingham.csv'):
    """
    Loads data, drops nulls, splits into train/test, and scales features.
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

if __name__ == "__main__":
    # This block only runs if you execute 'python data_prep.py' directly
    X_tr, X_te, y_tr, y_te, _, _ = prepare_data()
    print(f"Data Prep Test Passed. Training shape: {X_tr.shape}")