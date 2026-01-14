
# streamlit_app.py
# ----------------------------------------------------------
# Streamlit app for ML Assignment-2 (Binary Classification)
# - Model selection (6 models)
# - Metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
# - Confusion matrix & classification report
# - Test scoring on uploaded CSV (unlabeled) or repo test_data.csv
# ----------------------------------------------------------

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

# Try to import xgboost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ---------------------------
# Page config & header
# ---------------------------
st.set_page_config(page_title="ML Assignment 2 – Classification", layout="wide")
st.title("ML Assignment 2 · Classification · Streamlit App")
st.caption("Supports 6 classifiers, labeled metrics, and test-time predictions")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Settings")

model_options = [
    "Logistic Regression",
    "Decision Tree",
    "kNN",
    "Naive Bayes (Gaussian)",
    "Naive Bayes (Multinomial)",
    "Random Forest (Ensemble)"
]
if XGB_AVAILABLE:
    model_options.append("XGBoost (Ensemble)")

model_name = st.sidebar.selectbox("Choose Model", model_options)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.50, 0.01)
target_col = st.sidebar.text_input("Target Column (labeled data)", "Depression")
drop_cols_in = st.sidebar.text_input("Columns to drop (comma-separated)", "id,Name")

st.sidebar.markdown("---")
st.sidebar.write("**Upload data**")
uploaded_val = st.sidebar.file_uploader("Upload Labeled CSV (optional; for metrics)", type=["csv"])
uploaded_test = st.sidebar.file_uploader("Upload Test CSV (unlabeled; for predictions)", type=["csv"])

# ---------------------------
# Utility functions
# ---------------------------
CLASSIFICATION_METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

def build_preprocessor(X: pd.DataFrame, nb_variant: str = "") -> ColumnTransformer:
    """
    Preprocessing:
      - For MultinomialNB: numeric (median + MinMaxScaler) → non-negative
      - For others: numeric (median + StandardScaler)
      - Categorical: most_frequent + OneHotEncoder dense (to support GaussianNB)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    if nb_variant == "multinomial":
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])
    else:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    # Use dense one-hot for NB compatibility
    try:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    except TypeError:
        # For older scikit-learn (<1.2)
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "ConfusionMatrix": confusion_matrix(y_true, y_pred).tolist(),
        "ClassificationReport": classification_report(y_true, y_pred)
    }

def make_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=None, random_state=42)
    elif name == "kNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "Naive Bayes (Gaussian)":
        return GaussianNB()
    elif name == "Naive Bayes (Multinomial)":
        return MultinomialNB()
    elif name == "Random Forest (Ensemble)":
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    elif name == "XGBoost (Ensemble)" and XGB_AVAILABLE:
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="binary:logistic", eval_metric="logloss"
        )
    else:
        # Fallback if XGB not available but selected
        st.warning("XGBoost is unavailable. Falling back to Random Forest.")
        return RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")

def read_repo_train(train_path="train_data.csv"):
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    else:
        return None

def read_test_default(test_path="test_data.csv"):
    if os.path.exists(test_path):
        return pd.read_csv(test_path)
    else:
        return None

def parse_drop_cols(text: str):
    return [c.strip() for c in text.split(",") if c.strip()]

# ---------------------------
# Tabs: Metrics | Predict
# ---------------------------
tab_metrics, tab_predict = st.tabs(["📊 Metrics (Labeled)", "🔮 Predict (Test)"])

with tab_metrics:
    st.subheader("Validation Metrics")

    # Decide data source for metrics
    if uploaded_val is not None:
        df_val = pd.read_csv(uploaded_val)
        st.info("Using uploaded **labeled** CSV for metrics.")
    else:
        df_val = read_repo_train()
        if df_val is not None:
            st.info("Using repo **train_data.csv** for metrics.")
        else:
            st.warning("No labeled CSV provided and repo train_data.csv not found.")
            st.stop()

    drops = parse_drop_cols(drop_cols_in)
    if target_col not in df_val.columns:
        st.error(f"Target column '{target_col}' not found in the labeled dataset.")
        st.stop()

    # Prepare features & target
    X_val = df_val.drop(columns=[c for c in drops if c in df_val.columns] + [target_col])
    y_val = df_val[target_col].astype(int)

    nb_variant = "multinomial" if model_name == "Naive Bayes (Multinomial)" else ""
    preprocessor = build_preprocessor(X_val, nb_variant=nb_variant)
    model = make_model(model_name)
    clf = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Quick train on labeled data (demo)
    clf.fit(X_val, y_val)

    # Probabilities preferred
    try:
        y_proba = clf.predict_proba(X_val)[:, 1]
    except Exception:
        if hasattr(clf.named_steps["model"], "decision_function"):
            scores = clf.named_steps["model"].decision_function(clf.named_steps["preprocess"].transform(X_val))
            s_min, s_max = scores.min(), scores.max()
            y_proba = (scores - s_min) / (s_max - s_min + 1e-9)
        else:
            y_proba = clf.predict(X_val)

    # Metrics
    metrics = compute_metrics(y_val, y_proba, threshold=threshold)

    st.markdown("#### Metrics")
    st.json({k: metrics[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})

    st.markdown("#### Confusion Matrix")
    cm_df = pd.DataFrame(metrics["ConfusionMatrix"], index=["True 0","True 1"], columns=["Pred 0","Pred 1"])
    st.dataframe(cm_df)

    st.markdown("#### Classification Report")
    st.text(metrics["ClassificationReport"])

with tab_predict:
    st.subheader("Test Scoring (Unlabeled CSV)")

    # Decide test source
    if uploaded_test is not None:
        df_test = pd.read_csv(uploaded_test)
        st.info("Using uploaded **test** CSV for predictions.")
    else:
        df_test = read_test_default()
        if df_test is not None:
            st.info("Using repo **test_data.csv** for predictions.")
        else:
            st.info("Upload a test CSV (unlabeled) to get predictions.")
            st.stop()

    drops = parse_drop_cols(drop_cols_in)
    X_test = df_test.drop(columns=[c for c in drops if c in df_test.columns])

    # Prefer training on repo train_data.csv if available
    df_train_repo = read_repo_train()
    nb_variant = "multinomial" if model_name == "Naive Bayes (Multinomial)" else ""
    model = make_model(model_name)

    if df_train_repo is not None and target_col in df_train_repo.columns:
        X_train = df_train_repo.drop(columns=[c for c in drops if c in df_train_repo.columns] + [target_col])
        y_train = df_train_repo[target_col].astype(int)
        preprocessor = build_preprocessor(X_train, nb_variant=nb_variant)
        clf = Pipeline([("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)
    else:
        # Demo fallback (not ideal): fit on X_test with fake labels
        st.warning("Repo training file not found or missing target. Fitting a demo model on test data with fake labels (predictions are illustrative only).")
        preprocessor = build_preprocessor(X_test, nb_variant=nb_variant)
        clf = Pipeline([("preprocess", preprocessor), ("model", model)])
        y_fake = np.zeros(len(X_test))
        clf.fit(X_test, y_fake)

    # Predict probabilities on test
    try:
        test_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        if hasattr(clf.named_steps["model"], "decision_function"):
            s = clf.named_steps["model"].decision_function(clf.named_steps["preprocess"].transform(X_test))
            s_min, s_max = s.min(), s.max()
            test_proba = (s - s_min) / (s_max - s_min + 1e-9)
        else:
            test_proba = np.zeros(len(X_test))

    test_pred = (test_proba >= threshold).astype(int)

    id_col = "id" if "id" in df_test.columns else None
    out = pd.DataFrame({
        id_col if id_col else "row_id": df_test[id_col] if id_col else np.arange(len(df_test)),
        "Depression_Prob": test_proba,
        "Depression_Pred": test_pred
    })
    st.markdown("#### Predictions (first 30 rows)")
    st.dataframe(out.head(30))

    st.download_button(
        "Download predictions CSV",
        data=out.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
