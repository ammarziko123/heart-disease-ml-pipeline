import os, json
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_supervised_pipeline.joblib")
FEATURE_INFO = os.path.join(ARTIFACTS_DIR, "feature_info.json")

st.set_page_config(page_title="Heart Disease Risk — ML", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Risk — ML Demo")
st.write("Enter patient attributes to estimate heart disease risk (for education only).")

@st.cache_data
def load_feature_info():
    with open(FEATURE_INFO, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_model():
    return load(MODEL_PATH)

def main():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_INFO)):
        st.error("Model artifacts not found. Please train the model first (run `python train.py`).")
        st.stop()

    info = load_feature_info()
    model = load_model()

    # Build an input form using the original feature list
    cols = info["feature_order"]
    numeric = set(info["numeric"])
    categorical = set(info["categorical"])

    with st.form("input-form"):
        st.subheader("Patient inputs")
        values = {}
        for c in cols:
            if c in numeric:
                # reasonable defaults & ranges (can be adjusted per dataset)
                step = 1.0
                default = 0.0
                values[c] = st.number_input(f"{c}", value=default)
            else:
                values[c] = st.text_input(f"{c}", value="0")
        submitted = st.form_submit_button("Predict")

    if submitted:
        X = pd.DataFrame([values], columns=cols)
        try:
            proba = model.predict_proba(X)[:,1][0]
            pred = int(proba >= 0.5)
        except Exception:
            # Some estimators like SVM with probability=False (we set True, but safe‑guard)
            pred = int(model.predict(X)[0])
            proba = None

        st.markdown("---")
        st.subheader("Result")
        if proba is not None:
            st.metric("Estimated risk probability", f"{proba:.2%}")
        st.metric("Predicted class (1 = disease, 0 = healthy)", f"{pred}")

        st.caption("⚠️ Educational demo only. Not medical advice.")

if __name__ == "__main__":
    main()
