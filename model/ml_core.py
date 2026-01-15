import os
import warnings
warnings.filterwarnings("ignore")

import sys
import os

# ✅ Add workspace root to Python path for cloud deployment compatibility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from xgboost import XGBClassifier

# ---------------------------
# Config (edit as needed)
# ---------------------------
TRAIN_PATH   = "train_data.csv"         # labeled dataset
TEST_PATH    = "test_data.csv"          # unlabeled dataset (optional, for scoring)
TARGET       = "Depression"             # binary target column
DROP_COLS    = ["id", "Name"]           # identifiers / PII to drop from features
THRESHOLD    = 0.50                     # decision threshold for 0/1
NAIVE_TYPE   = "gaussian"               # "gaussian" or "multinomial"

# ---------------------------
# Utility functions
# ---------------------------
CLASSIFICATION_METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def build_preprocessor(X: pd.DataFrame, for_model: str = "") -> ColumnTransformer:
    """
    Create preprocessing:
      - For MultinomialNB: numeric (median + MinMaxScaler -> non-negative), categorical (most_frequent + OHE dense)
      - For others: numeric (median + StandardScaler), categorical (most_frequent + OHE dense)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    multinomial = "multinomial" in for_model.lower() or "multinomial" in for_model.lower()
    if "naive bayes" in for_model.lower():
        # Distinguish Gaussian vs Multinomial using the name you pass in the loop
        multinomial = "multinomial" in for_model.lower()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()) if multinomial else ("scaler", StandardScaler())
    ])

    # Ensure dense output for compatibility with GaussianNB
    try:
        cat_onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older scikit-learn fallback
        cat_onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", cat_onehot)
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])


def compute_metrics(y_true, y_proba, threshold=0.5):
    """Return all required metrics; also include confusion matrix & report."""
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

def get_models(naive_variant="gaussian"):
    """Return dict of models to evaluate."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="liblinear", class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=None, random_state=42
        ),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB() if naive_variant == "gaussian" else MultinomialNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        ),
    }
    models["XGBoost (Ensemble)"] = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="binary:logistic", eval_metric="logloss"
    )
    return models
    
if __name__ == "__main__":
# ---------------------------
# Load & basic rubric checks
# ---------------------------
df = pd.read_csv(TRAIN_PATH)
assert df.shape[0] >= 500, "Dataset must have >= 500 instances."
assert TARGET in df.columns, f"Target column '{TARGET}' not found in training data."
assert df.drop(columns=[TARGET]).shape[1] >= 12, "Dataset must have >= 12 features."

# ---------------------------
# Train/Validation split
# ---------------------------
drops = [c for c in DROP_COLS if c in df.columns]
X = df.drop(columns=drops + [TARGET])
y = df[TARGET].astype(int)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
models = get_models(naive_variant=NAIVE_TYPE)

# ---------------------------
# Fit, evaluate, collect metrics
# ---------------------------
results = {}
for name, model in models.items():
    preprocessor = build_preprocessor(X_tr, for_model=name)
    clf = Pipeline([("preprocess", preprocessor), ("model", model)])
    clf.fit(X_tr, y_tr)

    # Prefer probabilities; fallback to decision_function or labels if needed
    try:
        y_val_proba = clf.predict_proba(X_val)[:, 1]
    except Exception:
        if hasattr(clf.named_steps["model"], "decision_function"):
            df_score = clf.named_steps["model"].decision_function(clf.named_steps["preprocess"].transform(X_val))
            df_min, df_max = df_score.min(), df_score.max()
            y_val_proba = (df_score - df_min) / (df_max - df_min + 1e-9)
        else:
            y_val_proba = clf.predict(X_val)

    m = compute_metrics(y_val, y_val_proba, threshold=THRESHOLD)
    results[name] = {k: m[k] for k in CLASSIFICATION_METRICS}
    results[name]["ConfusionMatrix"] = m["ConfusionMatrix"]
    results[name]["ClassificationReport"] = m["ClassificationReport"]

# ---------------------------
# Save comparison table & reports
# ---------------------------
metrics_df = pd.DataFrame(results).T[CLASSIFICATION_METRICS].round(4)
os.makedirs("model", exist_ok=True)
metrics_df.to_csv("model/metrics_summary.csv", index=True)

with open("model/detailed_reports.txt", "w") as f:
    for name in results:
        f.write(f"\n\n## {name}\n")
        f.write(f"Confusion Matrix: {results[name]['ConfusionMatrix']}\n")
        f.write(f"{results[name]['ClassificationReport']}\n")

print("\n=== Validation Metrics Summary ===")
print(metrics_df.to_string())

# ---------------------------
# Pick best model by AUC, refit on ALL training data, score TEST
# ---------------------------
best_name = max(results.keys(), key=lambda n: results[n]["AUC"])
print(f"\nBest model by AUC: {best_name} (AUC={results[best_name]['AUC']:.4f})")

best_model = models[best_name]
best_preprocessor = build_preprocessor(X, for_model=best_name)
best_clf = Pipeline([("preprocess", best_preprocessor), ("model", best_model)])
best_clf.fit(X, y)  # refit on full training data

# If test file exists, score and save predictions
if os.path.exists(TEST_PATH):
    df_test = pd.read_csv(TEST_PATH)
    X_test = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns])

    try:
        test_proba = best_clf.predict_proba(X_test)[:, 1]
    except Exception:
        if hasattr(best_clf.named_steps["model"], "decision_function"):
            s = best_clf.named_steps["model"].decision_function(best_clf.named_steps["preprocess"].transform(X_test))
            s_min, s_max = s.min(), s.max()
            test_proba = (s - s_min) / (s_max - s_min + 1e-9)
        else:
            # fallback: zeros
            test_proba = np.zeros(len(X_test))

    test_pred = (test_proba >= THRESHOLD).astype(int)

    id_col = "id" if "id" in df_test.columns else None
    out = pd.DataFrame({
        id_col if id_col else "row_id": df_test[id_col] if id_col else np.arange(len(df_test)),
        "Depression_Prob": test_proba,
        "Depression_Pred": test_pred
    })
else:
    print(f"\nNo '{TEST_PATH}' found. Skipping test scoring.")
