# app.py
import warnings
warnings.filterwarnings("ignore")

import sys
import os

# Adds the current directory to the path so local modules are found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ✅ Import your functions from model/ml_core.py
from model.ml_core import build_preprocessor, get_models, compute_metrics

# ---------------------------
# App constants
# ---------------------------
CLASSIFICATION_METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
DEFAULT_TARGET = "Depression"
DEFAULT_DROP_COLS = "id,Name"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------
# Load deployed test dataset (Evaluator reference)
# ---------------------------
TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model",
    "test_data.csv"
)

test_df_deployed = None
if os.path.exists(TEST_DATA_PATH):
    try:
        test_df_deployed = pd.read_csv(TEST_DATA_PATH)
    except Exception as e:
        st.error(f"Bundled test_data.csv found but unreadable: {e}")

# ---------------------------
# UI helpers
# ---------------------------
def get_proba(clf: Pipeline, X_val: pd.DataFrame) -> np.ndarray:
    """Prefer predict_proba; fallback to decision_function normalized; last resort labels."""
    try:
        return clf.predict_proba(X_val)[:, 1]
    except Exception:
        model = clf.named_steps.get("model", None)
        pre = clf.named_steps.get("preprocess", None)
        if model is not None and hasattr(model, "decision_function") and pre is not None:
            scores = model.decision_function(pre.transform(X_val))
            smin, smax = scores.min(), scores.max()
            return (scores - smin) / (smax - smin + 1e-9)
        return clf.predict(X_val).astype(float)

def build_ui_models() -> dict:
    models_g = get_models(naive_variant="gaussian")
    models_m = get_models(naive_variant="multinomial")

    rename = {
        "Logistic Regression": "Logistic Regression",
        "Decision Tree": "Decision Tree Classifier",
        "kNN": "K-Nearest Neighbor Classifier",
        "Random Forest (Ensemble)": "Ensemble Model – Random Forest",
        "XGBoost (Ensemble)": "Ensemble Model – XGBoost",
    }

    ui_map = {}
    for k, v in models_g.items():
        if k == "Naive Bayes":
            continue
        ui_map[rename.get(k, k)] = v

    if "Naive Bayes" in models_g:
        ui_map["Naive Bayes Classifier – Gaussian"] = models_g["Naive Bayes"]
    if "Naive Bayes" in models_m:
        ui_map["Naive Bayes Classifier – Multinomial"] = models_m["Naive Bayes"]

    return ui_map

def fit_and_eval_single(X, y, model_name, model_obj, threshold):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pre = build_preprocessor(X_tr, for_model=model_name)
    clf = Pipeline([("preprocess", pre), ("model", model_obj)])
    clf.fit(X_tr, y_tr)
    proba = get_proba(clf, X_val)
    return compute_metrics(y_val, proba, threshold)

def compare_all_models(X, y, threshold):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    for name, model in build_ui_models().items():
        pre = build_preprocessor(X_tr, for_model=name)
        clf = Pipeline([("preprocess", pre), ("model", model)])
        clf.fit(X_tr, y_tr)
        proba = get_proba(clf, X_val)
        try:
            m = compute_metrics(y_val, proba, threshold)
        except Exception:
            m = {k: np.nan for k in CLASSIFICATION_METRICS}

        rows.append({"Model": name, **{k: m.get(k, np.nan) for k in CLASSIFICATION_METRICS}})

    df = pd.DataFrame(rows)
    return df.sort_values(by=["AUC", "F1"], ascending=False, na_position="last").reset_index(drop=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Classification UI", page_icon="🤖", layout="wide")
st.title("🤖 Classification UI (Using model/ml_core.py)")

with st.sidebar:
    st.subheader("⬇️ Download Test Dataset")
    if test_df_deployed is not None:
        csv_bytes = test_df_deployed.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download test_data.csv",
            data=csv_bytes,
            file_name="test_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("test_data.csv not bundled with app.")

    st.header("Upload Test Data (Evaluator)")
    uploaded_test_file = st.file_uploader("Upload test_data.csv", type=["csv"])

    st.header("Model & Threshold")
    ui_models = build_ui_models()
    model_name = st.selectbox("Select model", list(ui_models.keys()))
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.55, 0.05)

    st.header("Detail View")
    detail_view = st.radio("Show", ["Confusion Matrix", "Classification Report"])

st.markdown("---")

if uploaded_test_file is None:
    st.info("⬆️ Download and upload the test CSV to proceed.")
    st.stop()

# ---------------------------
# Load uploaded test data
# ---------------------------
df = pd.read_csv(uploaded_test_file)

# Verify integrity (optional but recommended)
if test_df_deployed is not None:
    if df.equals(test_df_deployed):
        st.success("✅ Uploaded test data matches deployed test_data.csv")
    else:
        st.warning("⚠️ Uploaded test data does NOT match deployed test_data.csv")

# ---------------------------
# Prepare X, y
# ---------------------------
if DEFAULT_TARGET not in df.columns:
    st.error(f"Target column '{DEFAULT_TARGET}' not found.")
    st.stop()

X = df.drop(columns=[DEFAULT_TARGET], errors="ignore")
y = df[DEFAULT_TARGET].astype(int)

# ---------------------------
# Tabs
# ---------------------------
tab_single, tab_compare = st.tabs(["🔹 Single Model", "🔸 Compare Models"])

with tab_single:
    model_obj = ui_models[model_name]
    metrics = fit_and_eval_single(X, y, model_name, model_obj, threshold)

    st.markdown("### 📊 Validation Metrics")
    st.dataframe(
        pd.DataFrame(metrics, index=[0]).T.rename(columns={0: "Value"}),
        use_container_width=True
    )

with tab_compare:
    st.markdown("### 🔍 Model Comparison")
    st.dataframe(compare_all_models(X, y, threshold), use_container_width=True)

st.markdown("---")
st.caption(
    "Test dataset is bundled with the app and made available for evaluator download and re-upload. "
    "Training logic remains unchanged and reproducible."
)
