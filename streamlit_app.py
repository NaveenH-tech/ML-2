

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

# --------------
# App constants
# --------------
CLASSIFICATION_METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
DEFAULT_TARGET = "Depression"
DEFAULT_DROP_COLS = "id,Name"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# -----------
# UI helpers 
# -----------
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
    """
    Use your get_models() to expose both NB variants and map to UI labels:
    - Logistic Regression
    - Decision Tree Classifier
    - K-Nearest Neighbor Classifier
    - Naive Bayes Classifier – Gaussian
    - Naive Bayes Classifier – Multinomial
    - Ensemble Model – Random Forest
    - Ensemble Model – XGBoost
    """
    models_g = get_models(naive_variant="gaussian")
    models_m = get_models(naive_variant="multinomial")

    rename = {
        "Logistic Regression": "Logistic Regression",
        "Decision Tree": "Decision Tree Classifier",
        "kNN": "K-Nearest Neighbor Classifier",
        "Random Forest (Ensemble)": "Ensemble Model – Random Forest",
        "XGBoost (Ensemble)": "Ensemble Model – XGBoost",
    }

    ui_map: dict = {}
    # Add non-NB models from the gaussian dict
    for k, v in models_g.items():
        if k == "Naive Bayes":
            continue
        ui_map[rename.get(k, k)] = v

    # Add NB variants explicitly
    if "Naive Bayes" in models_g:
        ui_map["Naive Bayes Classifier – Gaussian"] = models_g["Naive Bayes"]
    if "Naive Bayes" in models_m:
        ui_map["Naive Bayes Classifier – Multinomial"] = models_m["Naive Bayes"]

    return ui_map

def fit_and_eval_single(X, y, model_name: str, model_obj, threshold: float):
    """Fit/evaluate selected model on fixed 80/20 split using your functions."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pre = build_preprocessor(X_tr, for_model=model_name)  # your function
    clf = Pipeline([("preprocess", pre), ("model", model_obj)])
    clf.fit(X_tr, y_tr)
    proba = get_proba(clf, X_val)
    metrics = compute_metrics(y_val, proba, threshold=threshold)  # your function
    return metrics

def compare_all_models(X, y, threshold: float) -> pd.DataFrame:
    """Evaluate all models (including both NB variants) on the SAME split & threshold."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    ui_models = build_ui_models()
    for name, model in ui_models.items():
        pre = build_preprocessor(X_tr, for_model=name)
        clf = Pipeline([("preprocess", pre), ("model", model)])
        clf.fit(X_tr, y_tr)
        proba = get_proba(clf, X_val)
        try:
            m = compute_metrics(y_val, proba, threshold=threshold)
        except Exception:
            m = {k: np.nan for k in CLASSIFICATION_METRICS}
        rows.append({"Model": name, **{k: m.get(k, np.nan) for k in CLASSIFICATION_METRICS}})
    df = pd.DataFrame(rows)
    if df["AUC"].notna().any():
        df = df.sort_values(by=["AUC", "F1"], ascending=False, na_position="last")
    else:
        df = df.sort_values(by=["F1", "Accuracy"], ascending=False)
    return df.reset_index(drop=True)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Classification UI", page_icon="🤖", layout="wide")
st.title("🤖 Classification UI (Using model/ml_core.py)")

with st.sidebar:
    # ---------------------------
# Download Uploaded Dataset (for evaluator verification)
# ---------------------------
st.markdown("---")
    st.subheader("⬇️ Download Test Dataset")
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8") 
        st.download_button(
            label="Download Test Dataset (CSV)",
            data=csv_bytes,
            file_name="test_data.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Failed to prepare dataset for download: {e}")
   
    st.header("Upload Test Data")
    train_file = st.file_uploader("Test CSV (must include target)", type=["csv"])

    st.header("Columns")
    target_col = st.text_input("Target column (binary 0/1)", value=DEFAULT_TARGET, disabled=True )
    drop_cols_input = st.text_input("Columns to drop (comma-separated)", value=DEFAULT_DROP_COLS,disabled=True )
    drop_cols = [c.strip() for c in drop_cols_input.split(",") if c.strip()]

    st.header("Model & Threshold")
    ui_models = build_ui_models()
    model_name = st.selectbox("Select model", list(ui_models.keys()))
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.55, 0.05)

    st.header("Detail View")
    detail_view = st.radio("Show", ["Confusion Matrix", "Classification Report"])

st.markdown("---")

if train_file is None:
    st.info("⬆️ Upload a training CSV to get started.")
    st.stop()

# Load training data
try:
    df = pd.read_csv(train_file)
except Exception as e:
    st.error(f"Failed to read the uploaded CSV: {e}")
    st.stop()

# Validate target
if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in the uploaded CSV.")
    st.stop()

# Prepare X, y
drops = [c for c in drop_cols if c in df.columns]
X = df.drop(columns=drops + [target_col], errors="ignore")
y_raw = df[target_col]

# Coerce to binary int if possible
try:
    y = y_raw.astype(int)
except Exception:
    uniq = sorted(y_raw.dropna().unique())
    if len(uniq) == 2:
        mapping = {uniq[0]: 0, uniq[1]: 1}
        y = y_raw.map(mapping).astype(int)
        st.warning(f"Mapped target values {uniq} → {mapping}")
    else:
        st.error(f"Target '{target_col}' must be binary. Found unique values: {sorted(y_raw.unique())}")
        st.stop()

# Dataset info
col1, col2, col3 = st.columns(3)
with col1: st.metric("Rows", X.shape[0])
with col2: st.metric("Features", X.shape[1])
with col3:
    vc = pd.Series(y).value_counts(normalize=True).sort_index()
    st.metric("Class balance (0/1)", f"{vc.get(0,0):.2f} / {vc.get(1,0):.2f}")

# Tabs: Single Model | Compare Models
tab_single, tab_compare = st.tabs(["🔹 Single Model", "🔸 Compare Models"])

with tab_single:
    st.subheader("Selected Model")
    st.write(f"**Model:** {model_name}  |  **Threshold:** {threshold:.2f}  |  Split: 80/20 (random_state=42)")
    model_obj = ui_models[model_name]

    with st.spinner("Training & evaluating..."):
        try:
            metrics = fit_and_eval_single(X, y, model_name, model_obj, threshold)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # Metrics table
    st.markdown("### 📊 Validation Metrics")
    metrics_tbl = pd.DataFrame([{k: metrics.get(k, np.nan) for k in CLASSIFICATION_METRICS}]).T
    metrics_tbl.columns = ["Value"]
    st.dataframe(metrics_tbl.style.format("{:.4f}"), use_container_width=True)

    # Detailed view
    st.markdown("### 🔎 Detailed View")
    if detail_view == "Confusion Matrix":
        cm = np.array(metrics["ConfusionMatrix"])
        cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df.style.format("{:.0f}"), use_container_width=True)
    else:
        st.text(metrics["ClassificationReport"])

with tab_compare:
    st.subheader("Compare All Models")
    st.write(f"**Threshold:** {threshold:.2f}  |  Same split used for all models (80/20, random_state=42)")
    with st.spinner("Training & evaluating all models..."):
        try:
            comp_df = compare_all_models(X, y, threshold)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            st.stop()
    cols = ["Model"] + CLASSIFICATION_METRICS
    st.dataframe(comp_df[cols].style.format({c: "{:.4f}" for c in CLASSIFICATION_METRICS}), use_container_width=True)

st.markdown("---")
st.caption(
    "UI imports your functions from model/ml_core.py and does not re-implement core logic. "
    "A single 80/20 stratified split with random_state=42 is used consistently."
)
