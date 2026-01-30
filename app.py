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

# --- Main App Logic using Tabs ---
tab1, tab2 = st.tabs(["🎲 Generate Test Data", "🚀 Prediction Dashboard"])

# ==========================================
# TAB 1: DATA GENERATOR
# ==========================================
with tab1:
    st.header("🎲 Random Test Data Generator")
    st.markdown("Don't have a file? Generate a random sample from the original dataset to test the models.")
    
    # Check if source file exists
    source_file = "framingham_heart_study.csv"
    if os.path.exists(source_file):
        df_full = pd.read_csv(source_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Settings")
            # Slider for percentage
            sample_percent = st.slider("Select Data Percentage (%)", 1, 50, 10)
            
            # Randomize Button
            if st.button("🎲 Randomize & Sample"):
                # Sampling logic
                df_sample = df_full.sample(frac=sample_percent/100, random_state=None) # None for true random
                st.session_state['generated_data'] = df_sample
                st.success(f"Generated {len(df_sample)} random patient records!")
        
        with col2:
            st.subheader("Preview")
            if 'generated_data' in st.session_state:
                st.dataframe(st.session_state['generated_data'].head(8), use_container_width=True)
                
                # Download Button
                csv_data = st.session_state['generated_data'].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download This Sample CSV",
                    data=csv_data,
                    file_name="random_test_data.csv",
                    mime="text/csv",
                )
            else:
                st.info("👈 Click the 'Randomize' button to generate data.")
    else:
        st.warning("⚠️ Source 'framingham_heart_study.csv' not found. Cannot generate random samples.")

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

            # Handle Missing Values
            if X_input.isnull().values.any():
                st.warning("⚠️ Data contains missing values. Dropping incomplete rows...")
                if has_target:
                    data_temp = pd.concat([X_input, y_true], axis=1).dropna()
                    X_input = data_temp.drop(columns=[target_col])
                    y_true = data_temp[target_col]
                else:
                    X_input = X_input.dropna()

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
                    y_pred = model.predict(X_scaled)
                    
                    # Get probabilities if supported
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_scaled)[:, 1]
                    else:
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