import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="NASA Jet Engine RUL Predictor", 
    page_icon="✈️", 
    layout="wide"
)

st.title("✈️ Aircraft Engine Reliable RUL Predictor")
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
        
        if final_rul < 30:
            st.error("🚨 Critical Alert: Engine is near terminal degradation! Immediate maintenance required.")
        elif final_rul < 75:
            st.warning("⚠️ Warning: Engine shows moderate degradation. Schedule routine maintenance soon.")
        else:
            st.info("💚 Nominal Status: Engine is operating efficiently within safe boundaries.")
            
    except FileNotFoundError:
        st.error(f"Error: Trained model artifacts file `{artifacts_file}` not found. Please ensure it is uploaded to the directory.")