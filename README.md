# NASA Jet Engine Remaining Useful Life (RUL) Predictor ✈️

An enterprise-grade predictive maintenance application built with Python and Streamlit. This system leverages advanced machine learning models (XGBoost and LightGBM) to accurately predict the Remaining Useful Life (RUL) of aircraft turbofan engines based on the benchmark **NASA CMAPSS dataset**.

## 📊 Dataset Architectural Architecture
The application dynamically handles 4 distinct dataset constraints mapping different operational conditions and fault types:
- **FD001:** Single Condition - Single Fault Mode
- **FD002:** Multi-Condition - Single Fault Mode
- **FD003:** Single Condition - Multi-Fault Mode
- **FD004:** Multi-Condition - Multi-Fault Mode

## 🛠️ Data Pipeline & Feature Selection
To maintain peak generalization and prevent multicollinearity across diverse environments, the pipeline implements:
1. **Variance Threshold Filter:** Elimination of zero/quasi-constant dead sensor columns.
2. **PowerTransformer Processing:** Mitigation of skewness induced by highly dynamic operating settings.
3. **Correlation Profiling:** Dropping redundant parameters sharing correlation boundaries above `0.95`.
4. **Target Clipping:** Bounding targets to avoid early run-to-failure linear penalties.
