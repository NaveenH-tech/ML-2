
# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from model.utils import build_preprocessor, compute_metrics, DROP_DEFAULT

st.set_page_config(page_title="ML Assignment 2 - Classification App", layout="wide")
st.title("Machine Learning · Assignment 2 · Streamlit Demo")
st.caption("Dataset: train_data.csv (labeled), test_data.csv (unlabeled).")

# Sidebar controls
model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes (Gaussian)",
        "Naive Bayes (Multinomial)",
        "Random Forest (Ensemble)",
        "XGBoost (Ensemble)"
    ]
)
threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.50, 0.01)
drop_cols_in = st.sidebar.text_input("Columns to drop (comma-separated)", ",".join(DROP_DEFAULT))
target_col = st.sidebar.text_input("Target Column (for labeled validation)", "Depression")

uploaded_val = st.file_uploader("Upload VALIDATION CSV (must include target labels)", type=["csv"])
uploaded_test = st.file_uploader("Upload TEST CSV (no labels required)", type=["csv"])

def make_model(name):
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
    elif name == "XGBoost (Ensemble)":
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="binary:logistic", eval_metric="logloss"
        )

# Helper to read defaults if nothing uploaded
def read_default_train():
    # train_data has Depression labels (binary)
    return pd.read_csv("train_data.csv")  # labeled set [1](https://wilpbitspilaniacin0-my.sharepoint.com/personal/2025aa05847_wilp_bits-pilani_ac_in/Documents/Microsoft%20Copilot%20Chat%20Files/train_data.csv)

def read_default_test():
    # test_data typically has no Depression labels (unlabeled scoring)
    return pd.read_csv("test_data.csv")   # unlabeled set [2](https://wilpbitspilaniacin0-my.sharepoint.com/personal/2025aa05847_wilp_bits-pilani_ac_in/Documents/Microsoft%20Copilot%20Chat%20Files/test_data.csv)

with st.tabs(["Validation (metrics)", "Test Scoring (predictions)"])[0]:
    st.subheader("Validation (Labeled Data)")
    st.write("Upload a labeled CSV (or we’ll use `train_data.csv`) to compute metrics.")
    df_val = pd.read_csv(uploaded_val) if uploaded_val else read_default_train()
    drops = [c.strip() for c in drop_cols_in.split(",") if c.strip()]

    assert target_col in df_val.columns, "Target column not found in the validation dataset."
    X_val = df_val.drop(columns=[c for c in drops if c in df_val.columns] + [target_col])
    y_val = df_val[target_col].astype(int)

    preprocessor = build_preprocessor(X_val)
    model = make_model(model_name)
    clf = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Demo training: fit on the same labeled validation data (for metrics display)
    clf.fit(X_val, y_val)

    # Probabilities if available
    y_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_val)
    metrics = compute_metrics(y_val, y_proba, threshold=threshold)

    st.markdown("### Metrics")
    st.json({k: metrics[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]})

    st.markdown("### Confusion Matrix")
    st.write(pd.DataFrame(metrics["ConfusionMatrix"], index=["True 0","True 1"], columns=["Pred 0","Pred 1"]))

    st.markdown("### Classification Report")
    st.text(metrics["ClassificationReport"])

with st.tabs(["Validation (metrics)", "Test Scoring (predictions)"])[1]:
    st.subheader("Test Scoring (Unlabeled Data)")
    st.write("Upload an unlabeled CSV (or we’ll use `test_data.csv`) to generate predictions.")
    df_test = pd.read_csv(uploaded_test) if uploaded_test else read_default_test()
    drops = [c.strip() for c in drop_cols_in.split(",") if c.strip()]
    X_test = df_test.drop(columns=[c for c in drops if c in df_test.columns])

    preprocessor = build_preprocessor(X_test)
    model = make_model(model_name)
    clf = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Fit with dummy labels (demo) — for production, load a pre-trained model instead.
    y_fake = np.zeros(len(X_test))
    clf.fit(X_test, y_fake)

    proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else np.zeros(len(X_test))
    pred = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        "row_id": np.arange(len(X_test)),
        "Prob": proba,
        "Pred": pred
    })
    st.markdown("### Predictions (first 30 rows)")
    st.dataframe(out.head(30))
    st.download_button("Download predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv")
``
