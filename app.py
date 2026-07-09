import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(page_title="Turbofan Engine RUL", page_icon="✈️", layout="wide")
st.title("✈️ Turbofan Engine Remaining Useful Life (RUL) Predictor")

# ==========================================
# 2. Navigation
# ==========================================
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["📊 Dataset Overview & Analytics", "📈 Model Performance Evaluation", "🔮 Live RUL Predictor"])

# ==========================================
# PAGE 1: DATASET OVERVIEW & ANALYTICS
# ==========================================
if page == "📊 Dataset Overview & Analytics":
    st.header("📊 Dataset Overview & Feature Engineering Process")
    
    st.markdown("""
    ### 1. The C-MAPSS Dataset Overview
    The dataset consists of multiple multivariate time-series data from turbofan engines. 
    Each engine operates under different conditions and develops faults over time.
    """)
    
    summary_data = {
        "Dataset": ["FD001", "FD002", "FD003", "FD004"],
        "Train Engines": [100, 260, 100, 249],
        "Test Engines": [100, 259, 100, 248],
        "Operating Conditions": ["1 (Sea Level)", "6 (Various)", "1 (Sea Level)", "6 (Various)"],
        "Fault Modes": ["1 (HPC)", "1 (HPC)", "2 (HPC, Fan)", "2 (HPC, Fan)"]
    }
    st.table(pd.DataFrame(summary_data))
    
    st.markdown("---")
    st.subheader("🛠️ Data Preprocessing & Feature Selection Methodology")
    st.markdown("To ensure optimal model performance, a strict preprocessing pipeline was applied across all datasets:")
    
    st.markdown("#### Step 1: Handling Constant Features (Uniqueness)")
    st.write("Sensors with a constant value (unique value count = 1) across all cycles carry no predictive power and act as noise. These static operational settings and sensors were completely dropped from the training sets.")
    
    st.markdown("#### Step 2: Variance Thresholding")
    st.write("Features with extremely low variance (nearly constant) were eliminated using a `VarianceThreshold`. If a sensor's readings barely change during the engine's degradation, it is not useful for predicting the Remaining Useful Life (RUL).")
    
    st.markdown("#### Step 3: Correlation Analysis")
    st.write("We calculated the Pearson correlation coefficient between the remaining sensors and the `target_rul`. Features with an absolute correlation below a specific threshold (e.g., 0.1) were removed, keeping only the sensors most highly correlated with engine failure.")
    
    st.markdown("#### Step 4: Handling Skewness (Yeo-Johnson PowerTransformer)")
    st.write("Sensor readings in thermodynamic systems are highly skewed. We applied the `Yeo-Johnson PowerTransformer` instead of standard scaling. This transformation normalizes the variance and makes the data distribution more Gaussian (bell-shaped), which significantly improves Gradient Boosting performance.")

# ==========================================
# PAGE 2: MODEL PERFORMANCE EVALUATION
# ==========================================
elif page == "📈 Model Performance Evaluation":
    st.header("📈 Model Performance Evaluation")
    st.markdown("Select a dataset below to view the evaluation plots and error metrics for both XGBoost and LightGBM.")
    
    dataset_choice = st.selectbox("Select Dataset:", ["FD001", "FD002", "FD003", "FD004"])
    st.markdown("---")
    
    # ⚠️ قم بتعديل الأرقام هنا بناءً على نتائجك الحقيقية من الـ Notebook
    metrics_data = {
        "FD001": {"xgb_rmse": "46.74", "xgb_mae": "35.12", "lgb_rmse": "46.51", "lgb_mae": "34.80"},
        "FD002": {"xgb_rmse": "00.00", "xgb_mae": "00.00", "lgb_rmse": "00.00", "lgb_mae": "00.00"},
        "FD003": {"xgb_rmse": "00.00", "xgb_mae": "00.00", "lgb_rmse": "00.00", "lgb_mae": "00.00"},
        "FD004": {"xgb_rmse": "00.00", "xgb_mae": "00.00", "lgb_rmse": "00.00", "lgb_mae": "00.00"}
    }
    
    current_metrics = metrics_data[dataset_choice]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 XGBoost Results")
        # Ensure images are named exactly like this in your 'assets' folder
        img_path = f"assets/{dataset_choice.lower()}_xgb.png" 
        if os.path.exists(img_path):
            st.image(img_path, caption=f"XGBoost Predictions for {dataset_choice}", use_container_width=True)
        else:
            st.warning(f"Image not found. Please upload '{img_path}' to GitHub.")
            
        st.success(f"**RMSE:** {current_metrics['xgb_rmse']}")
        st.info(f"**MAE:** {current_metrics['xgb_mae']}")
        
    with col2:
        st.subheader("⚡ LightGBM Results")
        img_path = f"assets/{dataset_choice.lower()}_lgb.png"
        if os.path.exists(img_path):
            st.image(img_path, caption=f"LightGBM Predictions for {dataset_choice}", use_container_width=True)
        else:
            st.warning(f"Image not found. Please upload '{img_path}' to GitHub.")
            
        st.success(f"**RMSE:** {current_metrics['lgb_rmse']}")
        st.info(f"**MAE:** {current_metrics['lgb_mae']}")

# ==========================================
# PAGE 3: LIVE RUL PREDICTOR (Your Original Code)
# ==========================================
elif page == "🔮 Live RUL Predictor":
    st.header("🔮 Fleet Deployment: Real-time Telemetry Inference")
    st.write("Predict the RUL using advanced machine learning models trained on NASA's CMAPSS dataset.")

    condition_choice = st.selectbox(
        "Select Engine Operational Environment & Fault Complexity:",
        ["FD001", "FD002", "FD003", "FD004"]
    )

    file_mapping = {
        "FD001": "artifacts_fd001.pkl",
        "FD002": "artifacts_fd002.pkl",
        "FD003": "artifacts_fd003.pkl",
        "FD004": "artifacts_fd004.pkl"
    }
    artifacts_file = file_mapping[condition_choice]

    st.subheader("📥 Current Operational Settings & Sensor Telemetry")

    col1, col2, col3, col4 = st.columns(4)
    user_inputs = {}

    with col1:
        st.markdown("### ⚙️ Settings")
        for i in range(1, 4):
            user_inputs[f'op_setting_{i}'] = st.number_input(f'Setting {i}', value=0.0, format="%.4f")

    sensors = list(range(1, 22))
    for idx, s_num in enumerate(sensors):
        current_col = [col2, col3, col4][idx % 3]
        with current_col:
            if idx % 3 == 0 and idx < 3:
                st.markdown("### 📊 Sensors")
            user_inputs[f'sensor_{s_num}'] = st.number_input(f'Sensor {s_num}', value=0.0, format="%.4f")

    st.markdown("---")
    if st.button("🔮 Predict Remaining Useful Life", type="primary"):
        try:
            with open(artifacts_file, 'rb') as f:
                artifacts = pickle.load(f)
                
            model = artifacts['model']
            pt = artifacts['power_transformer']
            scaler = artifacts['scaler']
            selected_features = artifacts['selected_features']
            
            input_df = pd.DataFrame([user_inputs])
            input_filtered = input_df[selected_features]
            
            input_transformed = pt.transform(input_filtered)
            input_scaled = scaler.transform(input_transformed)
            
            prediction = model.predict(input_scaled)[0]
            final_rul = int(round(prediction))
            
            st.success("Analysis Completed Successfully!")
            st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{final_rul} Cycles")
            
        except FileNotFoundError:
            st.error(f"Error: Trained model artifacts file `{artifacts_file}` not found. Please upload it to GitHub.")