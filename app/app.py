import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib

st.set_page_config(page_title="Machine Failure Prediction", layout="wide")


# --- Load models (cached so this only runs once, not on every click) ---
@st.cache_resource
def load_models():
    binary_model = XGBClassifier()
    binary_model.load_model('binary_failure_model.json')

    multi_model = XGBClassifier()
    multi_model.load_model('failure_type_model.json')

    le = joblib.load('label_encoder.pkl')

    return binary_model, multi_model, le


binary_model, multi_model, le = load_models()


# --- Sidebar: model performance summary ---
st.sidebar.header("Model Performance")
st.sidebar.metric("Recall (Failure Detection)", "76%")
st.sidebar.metric("Precision", "80%")
st.sidebar.metric("Dataset Size", "10,000 machines")
st.sidebar.metric("Failure Rate", "3.39%")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** XGBoost with tuned decision threshold (0.7)")


# --- Title and key findings ---
st.title("Machine Failure Prediction")

with st.expander("Key Findings from Analysis"):
    st.markdown("""
    - **Torque** and **Tool wear** are the strongest individual predictors of failure
    - **Rotational speed** matters most in combination with other readings — not obvious from simple averages alone
    - **Heat Dissipation Failure (HDF)** is the most common failure type
    - Lower quality-tier machines (Type L) fail nearly **2x more often** than high-tier (Type H)
    - **Known limitation:** Random failures (RNF) are unpredictable by design; Tool Wear Failures (TWF) had too few examples (42) for reliable detection
    """)

    importance_df = pd.DataFrame({
        'Feature': ['Rotational speed', 'Torque', 'Tool wear', 'Air temperature',
                    'Type_M', 'Type_L', 'Process temperature', 'Type_H'],
        'Importance': [0.330, 0.250, 0.219, 0.057, 0.045, 0.043, 0.040, 0.014]
    }).sort_values('Importance', ascending=True)

    st.bar_chart(importance_df.set_index('Feature'))


# --- Input form ---
st.subheader("Enter sensor readings below to get a live prediction")

air_temp = st.slider("Air temperature (K)", 295.0, 305.0, 300.0)
process_temp = st.slider("Process temperature (K)", 305.0, 315.0, 310.0)
rotational_speed = st.slider("Rotational speed (rpm)", 1150, 2900, 1500)
torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0)
tool_wear = st.slider("Tool wear (min)", 0, 260, 100)
product_type = st.selectbox("Product Type", ["L", "M", "H"])


# --- Prediction ---
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Air temperature': [air_temp],
        'Process temperature': [process_temp],
        'Rotational speed': [rotational_speed],
        'Torque': [torque],
        'Tool wear': [tool_wear],
        'Type_H': [1 if product_type == 'H' else 0],
        'Type_L': [1 if product_type == 'L' else 0],
        'Type_M': [1 if product_type == 'M' else 0],
    })

    # Stage 1: binary prediction with tuned threshold
    fail_prob = binary_model.predict_proba(input_data)[:, 1][0]
    threshold = 0.7
    will_fail = fail_prob >= threshold

    st.subheader("Prediction Result")
    st.write(f"Failure probability: {fail_prob:.2%}")

    if will_fail:
        st.error("Machine predicted to FAIL")

        # Stage 2: exclude "No Failure" - we already know it's failing,
        # so we want the most likely REAL failure type
        probs = multi_model.predict_proba(input_data)[0]
        no_failure_idx = list(le.classes_).index("No Failure")
        probs[no_failure_idx] = -1

        failure_type_enc = probs.argmax()
        failure_type = le.inverse_transform([failure_type_enc])[0]
        st.info(f"Predicted failure type: **{failure_type}**")
    else:
        st.success("Machine predicted to operate normally")

    with st.expander("Debug: view model input"):
        st.dataframe(input_data)
