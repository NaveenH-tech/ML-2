import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "model/trained_models/all_models.pkl"
TARGET_COLUMN = "Depression"
THRESHOLD = 0.5

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

# -------------------------------------------------
# LOAD TRAINED MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model file not found")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

models = load_models()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def get_proba(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except Exception:
        return model.predict(X).astype(float)

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred),
        "ClassificationReport": classification_report(y_true, y_pred)
    }

    try:
        results["AUC"] = roc_auc_score(y_true, y_proba)
    except Exception:
        results["AUC"] = np.nan

    return results

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Classification",
    layout="wide"
)

st.title("Classification App - ML Assignment 2")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Upload Test Dataset")
    test_file = st.file_uploader(
        "Upload test CSV only",
        type=["csv"]
    )

    st.header("Model Selection")
    model_name = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    st.header("Decision Threshold")
    threshold = st.slider(
        "Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    st.header("Detailed Output")
    view_option = st.radio(
        "View",
        ["Confusion Matrix", "Classification Report"]
    )

# ---------------- Main ----------------
if test_file is None:
    st.info("Upload test dataset to begin")
    st.stop()

# Load test data
test_df = pd.read_csv(test_file)

if TARGET_COLUMN not in test_df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found in test data")
    st.stop()

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = test_df[TARGET_COLUMN].astype(int)

# Dataset info
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Samples", X_test.shape[0])
with c2:
    st.metric("Features", X_test.shape[1])
with c3:
    balance = y_test.value_counts(normalize=True).to_dict()
    st.metric("Class Balance (0 / 1)", f"{balance.get(0,0):.2f} / {balance.get(1,0):.2f}")

# Run model
model = models[model_name]

with st.spinner("Evaluating model"):
    proba = get_proba(model, X_test)
    metrics = compute_metrics(y_test, proba, threshold)

# ---------------- Metrics ----------------
st.subheader("Evaluation Metrics")

metric_df = pd.DataFrame(
    [{"Metric": k, "Value": metrics[k]} for k in METRICS]
)

st.dataframe(metric_df.style.format({"Value": "{:.4f}"}), use_container_width=True)

# ---------------- Detailed View ----------------
st.subheader("Detailed Evaluation")

if view_option == "Confusion Matrix":
    cm = metrics["ConfusionMatrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"]
    )
    st.dataframe(cm_df, use_container_width=True)
else:
    st.text(metrics["ClassificationReport"])

# ---------------- Download Section ----------------
st.markdown("---")
st.subheader("Download Test Dataset")

st.download_button(
    label="Download Uploaded Test Data",
    data=test_df.to_csv(index=False),
    file_name="test_data.csv",
    mime="text/csv"
)

st.caption(
    "Models are pre-trained offline and loaded from disk. "
    "Only test data is uploaded as per assignment guidelines."
)
