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
    page_title="Heart Disease Risk AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS Styling (Theme Aware) ---
st.markdown("""
    <style>
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Metrics Styling - Uses Theme Colors */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: var(--primary-color);
    }
    
    /* Button Styling */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    /* Success Message - Uses Theme Colors */
    .stSuccess {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model(filename):
    model_path = os.path.join("model", filename)
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def get_sample_data():
    """Loads test_framingham.csv from data folder."""
    # Check paths (robustness for local vs deployed env)
    possible_paths = [
        os.path.join("data", "test_framingham.csv"),
        "test_framingham.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
            
    return None

# --- App Header ---
col1, col2 = st.columns([1, 5])
with col1:
    # Use a transparent background icon or standard emoji to look good in dark mode
    st.markdown("<h1>❤️</h1>", unsafe_allow_html=True)
with col2:
    st.title("Coronary Heart Disease Prediction")
    st.markdown("##### 🏥 Advanced AI Diagnostic Tool")

st.markdown("---")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    model_options = [
        "Logistic Regression",
        "Decision Tree",
        "k-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
    
    selected_model_name = st.selectbox("Select Classifier", model_options)
    
    # Model Descriptions
    descriptions = {
        "Logistic Regression": "Best for interpretable baseline results. High Accuracy.",
        "Decision Tree": "Flowchart-like decision making. Easy to explain.",
        "k-Nearest Neighbors": "Predicts based on similar patient history.",
        "Naive Bayes": "Probabilistic model. Good for high sensitivity.",
        "Random Forest": "Ensemble of trees. High Precision.",
        "XGBoost": "Gradient boosting. High performance on complex data."
    }
    st.info(f"**Info:** {descriptions[selected_model_name]}")
    st.markdown("---")
    st.caption("v1.0.0 | BITS Pilani ML Assignment")

# --- Load Model ---
model_filenames = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "k-Nearest Neighbors": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

model_file = model_filenames[selected_model_name]
loaded_data = load_model(model_file)

if not loaded_data:
    st.error(f"⚠️ Model file `{model_file}` not found. Please train the models first.")
    st.stop()

model = loaded_data['model']
scaler = loaded_data['scaler']
feature_names = loaded_data['feature_names']
custom_threshold = loaded_data.get('threshold', 0.5)

# --- Main Content Area ---

# 1. Data Selection Panel
st.subheader("1. Select Patient Data")

data_source = st.radio("Choose Data Source:", 
                      ["☁️ Load Sample Data (Built-in)", "📂 Upload CSV File"], 
                      horizontal=True)

df = None

if "Sample" in data_source:
    # Load test data from data/ folder
    df = get_sample_data()
    if df is not None:
        st.success(f"✅ Loaded test data from `data/test_framingham.csv` ({len(df)} records)")
        
        # Allow user to download this sample
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download test_framingham.csv",
            data=csv,
            file_name="test_framingham.csv",
            mime="text/csv",
            help="Download this sample to use for manual testing"
        )
    else:
        st.error("Could not find 'data/test_framingham.csv'. Please check the repository structure.")

else:
    # Upload handler
    uploaded_file = st.file_uploader("Upload 'test_framingham.csv' or similar", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom File Uploaded Successfully")

# 2. Prediction Engine
if df is not None:
    st.markdown("---")
    st.subheader("2. AI Analysis Dashboard")
    
    with st.expander("🔍 View Raw Patient Data", expanded=False):
        st.dataframe(df.head(), use_container_width=True)

    # Preprocessing
    target_col = 'TenYearCHD'
    has_target = target_col in df.columns

    if has_target:
        X_input = df.drop(columns=[target_col])
        y_true = df[target_col]
    else:
        X_input = df
        y_true = None

    # Handle Missing
    if X_input.isnull().values.any():
        st.warning("⚠️ Data contains missing values. Dropping incomplete records.")
        if has_target:
            temp = pd.concat([X_input, y_true], axis=1).dropna()
            X_input = temp.drop(columns=[target_col])
            y_true = temp[target_col]
        else:
            X_input = X_input.dropna()

    # Align Columns & Scale
    try:
        X_input = X_input[feature_names]
        X_scaled = scaler.transform(X_input)
    except KeyError as e:
        st.error(f"❌ Input data error: {e}")
        st.stop()

    # Run Button
    col_run, col_dummy = st.columns([1, 3])
    with col_run:
        run_btn = st.button("⚡ Run Diagnosis", type="primary")

    if run_btn:
        with st.spinner('Analyzing patterns...'):
            # Predict Logic
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_scaled)[:, 1]
                y_pred = (y_prob >= custom_threshold).astype(int)
            else:
                y_pred = model.predict(X_scaled)
                y_prob = None

        # Display Alert for Custom Threshold
        if custom_threshold != 0.5:
            st.info(f"ℹ️ **Note:** Model tuned with sensitivity threshold: **{custom_threshold}**")

        # --- Metrics Row ---
        if has_target:
            st.markdown("### 📊 Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.1%}", delta_color="normal")
            m2.metric("Precision", f"{precision_score(y_true, y_pred):.1%}", delta_color="normal")
            m3.metric("Recall (Sensitivity)", f"{recall_score(y_true, y_pred):.1%}", delta_color="inverse") # Inverted because high is good
            m4.metric("F1 Score", f"{f1_score(y_true, y_pred):.1%}")

            # Charts
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                # Transparent background for dark mode compatibility
                fig.patch.set_alpha(0)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                plt.ylabel('Actual Condition')
                plt.xlabel('Predicted Condition')
                st.pyplot(fig)
            
            with col_graph2:
                if y_prob is not None:
                    st.markdown("**Risk Probability Distribution**")
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    fig2.patch.set_alpha(0)
                    sns.histplot(y_prob, bins=20, kde=True, color="#e74c3c", ax=ax2)
                    plt.axvline(custom_threshold, color='black', linestyle='--', label=f'Cutoff {custom_threshold}')
                    plt.legend()
                    st.pyplot(fig2)

        # --- Results Table ---
        st.markdown("### 📋 Detailed Predictions")
        results_df = X_input.copy()
        results_df["Predicted Risk"] = ["🔴 High Risk" if x == 1 else "🟢 Low Risk" for x in y_pred]
        if y_prob is not None:
            results_df["Confidence"] = y_prob
        
        st.dataframe(results_df[["Predicted Risk"] + (["Confidence"] if y_prob is not None else []) + feature_names].head(10), use_container_width=True)

else:
    st.info("👈 Please select a Data Source to start.")