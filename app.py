import streamlit as st
import pandas as pd
import numpy as np
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
    
    /* Metrics Card Styling */
    .metric-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 15px;
        background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent white */
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
        margin: 5px;
        height: 100%;
        text-align: center;
    }

    /* Metric Label */
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        opacity: 0.8;
        margin-bottom: 5px;
        color: var(--text-color);
    }

    /* Metric Value */
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
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
    
    /* Center Plot Titles */
    .plot-title {
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 1.1rem;
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

def metric_card(label, value, color_hex="#4e73df"):
    """Displays a custom styled metric card."""
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color_hex};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# --- App Header ---
# Using Custom HTML/Flexbox to control the gap precisely
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
    <div style="font-size: 3rem;">❤️</div>
    <div>
        <h1 style="margin: 0; padding: 0;">Coronary Heart Disease Prediction</h1>
        <h5 style="margin: 0; padding: 0; opacity: 0.8;">🏥 Framingham Heart Study Analysis Tool</h5>
    </div>
</div>
""", unsafe_allow_html=True)

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
imputation_values = loaded_data.get('imputation_values', {})

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

    # Handle Missing Values (impute using training medians, matching training pipeline)
    if X_input.isnull().values.any():
        missing_count = X_input.isnull().sum().sum()
        if imputation_values:
            for col in X_input.columns:
                if col in imputation_values and X_input[col].isnull().any():
                    X_input[col] = X_input[col].fillna(imputation_values[col])
            st.info(f"ℹ️ Imputed {missing_count} missing values using training medians (all {len(X_input)} records preserved).")
        else:
            st.warning("⚠️ Data contains missing values. Dropping incomplete records (no imputation values available).")
            if has_target:
                temp = pd.concat([X_input, y_true], axis=1).dropna()
                X_input = temp.drop(columns=[target_col])
                y_true = temp[target_col]
            else:
                X_input = X_input.dropna()

    # Feature Engineering (same transformations as training)
    X_input = engineer_features(X_input)

    # Align Columns & Scale
    try:
        X_input = X_input[feature_names]
        X_scaled = scaler.transform(X_input)
    except KeyError as e:
        st.error(f"❌ Input data error: {e}. Ensure input CSV has the same columns as the training data.")
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
            
            # Calculate all required metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0

            # Display in Cards
            # Define colors for different metrics
            colors = {
                "Accuracy": "#4e73df",
                "AUC": "#f6c23e",
                "Precision": "#1cc88a",
                "Recall": "#e74c3c", 
                "F1": "#36b9cc",
                "MCC": "#858796"
            }
            
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            
            with m1: metric_card("Accuracy", f"{acc:.1%}", colors["Accuracy"])
            with m2: metric_card("AUC Score", f"{auc:.3f}" if y_prob is not None else "N/A", colors["AUC"])
            with m3: metric_card("Precision", f"{prec:.1%}", colors["Precision"])
            with m4: metric_card("Recall", f"{rec:.1%}", colors["Recall"])
            with m5: metric_card("F1 Score", f"{f1:.3f}", colors["F1"])
            with m6: metric_card("MCC Score", f"{mcc:.3f}", colors["MCC"])

            st.markdown("<br>", unsafe_allow_html=True) # Spacer

            # Charts
            col_graph1, col_graph2 = st.columns(2)
            
            with col_graph1:
                st.markdown('<div class="plot-title">Confusion Matrix</div>', unsafe_allow_html=True)
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                # Transparent background for dark mode compatibility
                fig.patch.set_alpha(0)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={"size": 14})
                plt.ylabel('Actual Condition (0=No, 1=Yes)', fontsize=10)
                plt.xlabel('Predicted Condition (0=No, 1=Yes)', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col_graph2:
                # Production Grade Bar Chart
                
                # Prepare data for plotting
                metrics_data = {
                    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                    "Value": [acc, auc, prec, rec, f1, mcc]
                }
                metrics_df_plot = pd.DataFrame(metrics_data)
                
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                # Transparent background to blend with theme
                fig2.patch.set_alpha(0)
                fig2.patch.set_facecolor('white')
                ax2.set_facecolor('white')
                
                # Create Horizontal Bar Plot for better label readability
                bar_plot = sns.barplot(
                    data=metrics_df_plot, 
                    y="Metric", 
                    x="Value", 
                    hue="Metric", 
                    palette="viridis", 
                    ax=ax2,
                    legend=False,
                    orient='h'
                )
                
                # Add value labels
                for p in bar_plot.patches:
                    width = p.get_width()
                    if width > 0: 
                         ax2.text(
                            width + 0.02, 
                            p.get_y() + p.get_height()/2, 
                            f'{width:.2f}', 
                            ha="left", 
                            va="center", 
                            fontsize=10,
                            fontweight='bold'
                        )
                
                # Styling the plot
                ax2.set_title("Model Performance Summary", fontsize=14, fontweight='bold', pad=15)
                ax2.set_xlim(0, 1.2) # Add headroom for labels
                ax2.set_xlabel("Score", fontsize=11, fontweight='bold')
                ax2.set_ylabel("Metric", fontsize=11, fontweight='bold')
                ax2.grid(axis='x', linestyle='--', alpha=0.3)
                
                # Remove top and right spines
                sns.despine(ax=ax2, left=True, bottom=False)
                
                plt.tight_layout()
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