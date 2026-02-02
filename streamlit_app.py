# streamlit_app.py
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
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "model/trained_models/all_models.pkl"
TEST_DATA_PATH = "data/test_data.csv"
DEFAULT_TARGET = "Depression"

# -------------------------------------------------
# LOAD PRETRAINED MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model file not found")
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


models = load_models()

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")
st.title("Classification App (Pretrained Models)")

# -------------------------------------------------
# DOWNLOAD TEST DATA (ALWAYS VISIBLE)
# -------------------------------------------------
st.subheader("Download Test Dataset")

if os.path.exists(TEST_DATA_PATH):
    test_df_download = pd.read_csv(TEST_DATA_PATH)
    csv_bytes = test_df_download.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Test Dataset (CSV)",
        data=csv_bytes,
        file_name="test_data.csv",
        mime="text/csv"
    )
else:
    st.warning("Test dataset file not found in data/test_data.csv")

st.markdown("---")

# -------------------------------------------------
# UPLOAD TEST DATA
# -------------------------------------------------
st.sidebar.header("1. Upload Test Dataset")
test_file = st.sidebar.file_uploader(
    "Upload CSV file (test data only)",
    type=["csv"]
)

st.sidebar.header("2. Settings")
target_col = st.sidebar.text_input("Target column", value=DEFAULT_TARGET, disabled=True )
model_name = st.sidebar.selectbox("Select model", list(models.keys()))
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05)
view_type = st.sidebar.radio(
    "Detailed View",
    ["Confusion Matrix", "Classification Report"]
)

# -------------------------------------------------
# STOP IF NO TEST DATA UPLOADED
# -------------------------------------------------
if test_file is None:
    st.info("Please upload the downloaded test CSV to view results.")
    st.stop()

# -------------------------------------------------
# LOAD TEST DATA
# -------------------------------------------------
df = pd.read_csv(test_file)

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in uploaded dataset")
    st.stop()

X_test = df.drop(columns=[target_col])
y_test = df[target_col].astype(int)

# -------------------------------------------------
# DATASET INFO
# -------------------------------------------------
st.markdown("### Dataset Information")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", X_test.shape[0])
c2.metric("Features", X_test.shape[1])
c3.metric(
    "Class Balance (0 / 1)",
    f"{(y_test == 0).mean():.2f} / {(y_test == 1).mean():.2f}"
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
model = models[model_name]

try:
    y_prob = model.predict_proba(X_test)[:, 1]
except Exception:
    y_prob = model.predict(X_test).astype(float)

y_pred = (y_prob >= threshold).astype(int)

# -------------------------------------------------
# METRICS
# -------------------------------------------------
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

try:
    metrics["AUC"] = roc_auc_score(y_test, y_prob)
except Exception:
    metrics["AUC"] = np.nan

st.markdown("### Evaluation Metrics")
metrics_df = pd.DataFrame(metrics, index=["Value"]).T
st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

# -------------------------------------------------
# DETAILED VIEW
# -------------------------------------------------
st.markdown("### Detailed Evaluation")

if view_type == "Confusion Matrix":
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["True 0", "True 1"],
        columns=["Pred 0", "Pred 1"]
    )
    st.dataframe(cm_df, use_container_width=True)
else:
    st.text(classification_report(y_test, y_pred))

st.caption(
    "Models are pretrained and loaded from disk. "
    "No model training is performed in the UI."
)
