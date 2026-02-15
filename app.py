import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)
from data_prep import engineer_features

# --- Page Config ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Title and Description ---
st.title("❤️ Coronary Heart Disease Prediction")
st.markdown("""
**Welcome!** This application predicts the 10-year risk of Coronary Heart Disease (CHD). 
You can generate random test scenarios or upload your own medical data to test various Machine Learning models.
""")

# --- Sidebar: Model Selection ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")
model_options = [
    "Logistic Regression",
    "Decision Tree",
    "k-Nearest Neighbors",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]

selected_model_name = st.sidebar.selectbox("🤖 Select ML Model", model_options)
st.sidebar.info(f"Currently using: **{selected_model_name}**")

# Map friendly names to filenames
model_filenames = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "k-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

# --- Helper Function to Load Model ---
@st.cache_resource
def load_model(filename):
    model_path = os.path.join("model", filename)
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

# --- Load the Selected Model ---
model_file = model_filenames[selected_model_name]
loaded_data = load_model(model_file)

if not loaded_data:
    st.error(f"⚠️ Model file `{model_file}` not found. Please train the models first.")
    st.stop()

model = loaded_data['model']
scaler = loaded_data['scaler']
feature_names = loaded_data['feature_names']
imputation_values = loaded_data.get('imputation_values', {})
saved_threshold = loaded_data.get('threshold', 0.5)  # Use calibrated threshold

# --- Main App Logic using Tabs ---
tab1, tab2 = st.tabs(["🎲 Generate Test Data", "🚀 Prediction Dashboard"])

# ==========================================
# TAB 1: DATA GENERATOR
# ==========================================
with tab1:
    st.header("🎲 Test Data Management")
    st.markdown("Generate random test data or load from saved files to evaluate your models.")
    
    # Sub-tabs for different options
    gen_tab1, gen_tab2 = st.tabs(["📊 Generate New Data", "📂 Load Saved Files"])
    
    # ==== GENERATE NEW DATA TAB ====
    with gen_tab1:
        source_file = "data/framingham_heart_study.csv"
        if os.path.exists(source_file):
            df_full = pd.read_csv(source_file)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("⚙️ Generation Settings")
                
                # Option to sample or create synthetic
                gen_method = st.radio("Generation Method:", 
                    ["Sample from Original", "Custom Random Range"], 
                    help="Sample: Random rows from original dataset\nCustom: Generate based on column statistics")
                
                if gen_method == "Sample from Original":
                    sample_percent = st.slider("Select Data Percentage (%)", 1, 50, 10)
                    sample_size = int(len(df_full) * sample_percent / 100)
                else:
                    sample_size = st.number_input("Number of Samples to Generate", 10, 500, 50)
                
                # Include target or not
                include_target = st.checkbox("Include Target Column (TenYearCHD)", value=True,
                    help="Check to test model performance, uncheck to test predictions only")
                
                # Randomize Button
                if st.button("🎲 Generate Data", type="primary"):
                    with st.spinner("Generating test data..."):
                        if gen_method == "Sample from Original":
                            df_sample = df_full.sample(n=min(sample_size, len(df_full)), random_state=None)
                        else:
                            # Generate synthetic data based on original statistics
                            import numpy as np
                            df_sample = pd.DataFrame()
                            for col in df_full.columns:
                                col_mean = df_full[col].mean()
                                col_std = df_full[col].std()
                                df_sample[col] = np.random.normal(col_mean, col_std, sample_size)
                            # Round numeric columns appropriately
                            for col in df_sample.columns:
                                if 'Age' in col or col == 'TenYearCHD':
                                    df_sample[col] = df_sample[col].astype(int)
                                else:
                                    df_sample[col] = df_sample[col].round(2)
                        
                        # Drop target if not needed
                        if not include_target:
                            df_sample = df_sample.drop(columns=['TenYearCHD'], errors='ignore')
                        
                        st.session_state['generated_data'] = df_sample
                        st.success(f"✅ Generated {len(df_sample)} test records!")
            
            with col2:
                st.subheader("👀 Data Preview")
                if 'generated_data' in st.session_state:
                    st.dataframe(st.session_state['generated_data'].head(10), use_container_width=True)
                    
                    # Download Button
                    csv_data = st.session_state['generated_data'].to_csv(index=False).encode('utf-8')
                    
                    # Create filename with timestamp
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"test_data_{timestamp}.csv"
                    
                    st.download_button(
                        label="📥 Download Test Data CSV",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Info box
                    st.info(f"📊 Shape: {st.session_state['generated_data'].shape[0]} rows × {st.session_state['generated_data'].shape[1]} columns")
                else:
                    st.info("👈 Click 'Generate Data' button to create test data.")
        else:
            st.warning("⚠️ Source 'framingham_heart_study.csv' not found. Cannot generate samples.")
    
    # ==== LOAD SAVED FILES TAB ====
    with gen_tab2:
        st.subheader("📂 Load Saved Test Files")
        
        # Option 1: Browse test_data folder
        test_data_folder = "test_data"
        
        # Create folder if it doesn't exist
        if not os.path.exists(test_data_folder):
            os.makedirs(test_data_folder)
        
        # List CSV files in test_data folder
        csv_files = [f for f in os.listdir(test_data_folder) if f.endswith('.csv')]
        
        if csv_files:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.write("**Available Test Files:**")
                selected_file = st.selectbox(
                    "Select a test file",
                    csv_files,
                    label_visibility="collapsed"
                )
                
                if st.button("📂 Load Selected File", type="primary"):
                    file_path = os.path.join(test_data_folder, selected_file)
                    try:
                        df_loaded = pd.read_csv(file_path)
                        st.session_state['generated_data'] = df_loaded
                        st.success(f"✅ Loaded {selected_file}")
                    except Exception as e:
                        st.error(f"❌ Error loading file: {e}")
            
            with col2:
                st.write("**File Details:**")
                for filename in csv_files:
                    filepath = os.path.join(test_data_folder, filename)
                    file_size = os.path.getsize(filepath) / 1024  # KB
                    file_rows = len(pd.read_csv(filepath))
                    st.caption(f"📄 {filename} ({file_rows} rows, {file_size:.1f} KB)")
        else:
            st.info(f"📁 No test files found in '{test_data_folder}' folder.\n\n"
                   f"**Getting Started:**\n"
                   f"1. Generated test data using the 'Generate New Data' tab\n"
                   f"2. Download the CSV file\n"
                   f"3. Move it to the '{test_data_folder}' folder\n"
                   f"4. Refresh this tab to see it here!")

# ==========================================
# TAB 2: PREDICTION DASHBOARD
# ==========================================
with tab2:
    st.header("🚀 Run Predictions")
    
    # File Upload Section
    uploaded_file = st.file_uploader("📂 Upload your CSV file (or the one you just downloaded)", type="csv")

    if uploaded_file is not None:
        try:
            # Read Data
            df = pd.read_csv(uploaded_file)
            
            # Layout: Data Info
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Target Column", "Present ✅" if 'TenYearCHD' in df.columns else "Missing ❌")

            with st.expander("👀 View Raw Data"):
                st.dataframe(df.head())

            # Preprocessing Logic
            target_col = 'TenYearCHD'
            has_target = target_col in df.columns

            # Drop target if present
            if has_target:
                X_input = df.drop(columns=[target_col])
                y_true = df[target_col]
            else:
                X_input = df
                y_true = None

            # Handle Missing Values (impute using training medians)
            if X_input.isnull().values.any():
                if imputation_values:
                    st.info("Imputing missing values using training data medians...")
                    X_input = X_input.fillna(imputation_values)
                    if has_target and y_true is not None:
                        # Keep alignment
                        mask = ~df.drop(columns=[target_col]).isnull().all(axis=1)
                        y_true = y_true[mask].reset_index(drop=True)
                        X_input = X_input[mask].reset_index(drop=True)
                else:
                    st.warning("\u26a0\ufe0f Data contains missing values. Dropping incomplete rows...")
                    if has_target:
                        data_temp = pd.concat([X_input, y_true], axis=1).dropna()
                        X_input = data_temp.drop(columns=[target_col])
                        y_true = data_temp[target_col]
                    else:
                        X_input = X_input.dropna()

            # Feature Engineering (same as training pipeline)
            X_input = engineer_features(X_input)

            # Feature Scaling
            try:
                # Ensure correct column order
                X_input = X_input[feature_names]
                X_scaled = scaler.transform(X_input)
            except KeyError as e:
                st.error(f"❌ Missing columns: {e}")
                st.stop()

            # Run Prediction Button
            if st.button("⚡ Run Model Prediction", type="primary"):
                with st.spinner('Analyzing patient data...'):
                    # Use saved threshold for probability-based prediction
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_scaled)[:, 1]
                        y_pred = (y_prob >= saved_threshold).astype(int)
                    else:
                        y_pred = model.predict(X_scaled)
                        y_prob = None

                # --- Results Display ---
                st.markdown("### 📊 Results Analysis")
                
                # Create results dataframe
                results_df = X_input.copy()
                results_df["Predicted_Risk"] = y_pred
                results_df["Risk_Label"] = ["High Risk 🚨" if x == 1 else "Low Risk ✅" for x in y_pred]
                
                if y_prob is not None:
                    results_df["Risk_Probability"] = y_prob

                # Show Predictions
                st.dataframe(results_df[["Predicted_Risk", "Risk_Label"] + (["Risk_Probability"] if y_prob is not None else [])].head(10), use_container_width=True)
                
                # Download Results
                res_csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Full Prediction Results",
                    res_csv,
                    "prediction_results.csv",
                    "text/csv"
                )

                # --- Metrics Section (Only if Ground Truth exists) ---
                if has_target:
                    st.markdown("---")
                    st.subheader("📈 Model Performance on Uploaded Data")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                    m2.metric("Precision", f"{precision_score(y_true, y_pred):.2%}")
                    m3.metric("Recall", f"{recall_score(y_true, y_pred):.2%}")
                    m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.2%}")

                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        st.caption("Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)
                    
                    with col_chart2:
                        if y_prob is not None:
                            st.caption("Risk Probability Distribution")
                            fig2, ax2 = plt.subplots(figsize=(4, 3))
                            sns.histplot(y_prob, bins=20, kde=True, color="orange", ax=ax2)
                            plt.xlabel("Predicted Probability")
                            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("👆 Please upload a CSV file to begin analysis.")