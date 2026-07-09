import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# 1. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="Turbofan Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Turbofan Engine Remaining Useful Life (RUL) Dashboard")
st.markdown("""
This production-ready application provides a comprehensive overview of the C-MAPSS dataset analytics, 
compares advanced Gradient Boosting models, and deploys live predictive models.
""")

# 2. القائمة الجانبية للتنقل بين الصفحات
st.sidebar.header("Navigation Menu")
page = st.sidebar.radio("Go to:", ["📊 Dataset Overview & Analytics", "📈 Model Performance Evaluation", "🔮 Live RUL Predictor"])

# =========================================================================
# PAGE 1: DATASET OVERVIEW & ANALYTICS
# =========================================================================
if page == "📊 Dataset Overview & Analytics":
    st.header("📊 C-MAPSS Dataset Statistical Profiling")
    st.subheader("Dataset Architecture Overview")
    summary_data = {
        "Dataset": ["FD001", "FD002", "FD003", "FD004"],
        "Operating Conditions": ["Single (1)", "Multi (6)", "Single (1)", "Multi (6)"],
        "Fault Modes": ["Single (HPC Degradation)", "Single (HPC Degradation)", "Multi (HPC + Fan)", "Multi (HPC + Fan)"],
        "Train Engines": [100, 260, 100, 249],
        "Test Engines": [100, 259, 100, 248]
    }
    st.table(pd.DataFrame(summary_data))
    
    st.markdown("### Dynamic Data Inspection")
    st.info("Typical Sensor Matrix Summary Statistics:")
    sample_df = pd.DataFrame(np.random.randn(10, 5), columns=['Sensor_2', 'Sensor_3', 'Sensor_4', 'Sensor_7', 'Sensor_11'])
    st.dataframe(sample_df.describe())

# =========================================================================
# PAGE 2: MODEL PERFORMANCE EVALUATION
# =========================================================================
elif page == "📈 Model Performance Evaluation":
    st.header("📈 Advanced Model Comparison Metrics")
    st.markdown("A direct comparison of baseline Tree models against our optimized Gradient Boosting implementations.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Random Forest Baseline RMSE", value="45.76")
    with col2:
        st.metric(label="XGBoost Regressor RMSE", value="46.74")
    with col3:
        st.metric(label="LightGBM Regressor RMSE", value="46.51")
        
    st.markdown("---")
    st.subheader("Visualizing Actual vs. Predicted RUL Trajectories")
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("#### XGBoost Regression Fit")
        if os.path.exists("assets/xgboost_eval.png"):
            st.image("assets/xgboost_eval.png", caption="XGBoost Actual vs Predicted")
        else:
            st.warning("Save your XGBoost plot in 'assets/xgboost_eval.png' to display it here.")
            
    with col_img2:
        st.markdown("#### LightGBM Regression Fit")
        if os.path.exists("assets/lgbm_eval.png"):
            st.image("assets/lgbm_eval.png", caption="LightGBM Actual vs Predicted")
        else:
            st.warning("Save your LightGBM plot in 'assets/lgbm_eval.png' to display it here.")

# =========================================================================
# PAGE 3: LIVE RUL PREDICTOR (كودك الأصلي الديناميكي)
# =========================================================================
elif page == "🔮 Live RUL Predictor":
    st.header("🔮 Fleet Deployment: Real-time Telemetry Inference")
    st.write("Predict the Remaining Useful Life (RUL) of turbofan engines using advanced machine learning models trained on NASA's CMAPSS dataset.")

    condition_choice = st.selectbox(
        "Select Engine Operational Environment & Fault Complexity:",
        [
            "FD001: Single Condition - Single Fault Mode",
            "FD002: Multi-Condition - Single Fault Mode",
            "FD003: Single Condition - Multi-Fault Mode",
            "FD004: Multi-Condition - Multi-Fault Mode"
        ]
    )

    file_mapping = {
        "FD001: Single Condition - Single Fault Mode": "artifacts_fd001.pkl",
        "FD002: Multi-Condition - Single Fault Mode": "artifacts_fd002.pkl",
        "FD003: Single Condition - Multi-Fault Mode": "artifacts_fd003.pkl",
        "FD004: Multi-Condition - Multi-Fault Mode": "artifacts_fd004.pkl"
    }
    artifacts_file = file_mapping[condition_choice]

    st.subheader("📥 Current Operational Settings & Sensor Telemetry")

    col1, col2, col3, col4 = st.columns(4)
    user_inputs = {}

    with col1:
        st.markdown("### ⚙️ Operating Settings")
        for i in range(1, 4):
            user_inputs[f'op_setting_{i}'] = st.number_input(f'Setting {i}', value=0.0, format="%.4f")

    sensors = list(range(1, 22))
    for idx, s_num in enumerate(sensors):
        current_col = [col2, col3, col4][idx % 3]
        with current_col:
            if idx % 3 == 0 and idx < 3:
                st.markdown("### 📊 Sensor Readings")
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
            
            # فلترة المدخلات بناءً على الفيتشرز المهمة اللي الموديل اتدرب عليها
            input_df = pd.DataFrame([user_inputs])
            input_filtered = input_df[selected_features]
            
            # تطبيق التحويلات
            input_transformed = pt.transform(input_filtered)
            input_scaled = scaler.transform(input_transformed)
            
            # التوقع
            prediction = model.predict(input_scaled)[0]
            final_rul = int(round(prediction))
            
            st.success("Analysis Completed Successfully!")
            st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{final_rul} Cycles")
            
            if final_rul < 30:
                st.error("🚨 Critical Alert: Engine is near terminal degradation! Immediate maintenance required.")
            elif final_rul < 75:
                st.warning("⚠️ Warning: Engine shows moderate degradation. Schedule routine maintenance soon.")
            else:
                st.info("💚 Nominal Status: Engine is operating efficiently within safe boundaries.")
                
        except FileNotFoundError:
            st.error(f"Error: Trained model artifacts file `{artifacts_file}` not found. Please ensure it is uploaded to the repository.")