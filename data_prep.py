import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv('framingham_heart_study.csv')

print(f"Original shape: {df.shape}")

# 2. Handle Missing Values
# Dropping rows with any NaN values
df_clean = df.dropna()
print(f"Shape after dropping nulls: {df_clean.shape}")

# 3. Define Features (X) and Target (y)
# The target for Framingham is 'TenYearCHD'
target_col = 'TenYearCHD'
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# 4. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData Preparation Complete.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Features count: {X_train.shape[1]}") # Should be > 12