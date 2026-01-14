
# model/train_and_eval.py
import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import build_preprocessor, compute_metrics, CLASSIFICATION_METRICS, DROP_DEFAULT

"""
Usage (BITS Virtual Lab):
python model/train_and_eval.py --data train_data.csv --target Depression --drop id Name --naive gaussian
"""

def load_data(path):
    df = pd.read_csv(path)
    return df

def split_X_y(df, target, drops):
    drops = [c for c in drops if c in df.columns]
    X = df.drop(columns=drops + [target])
    y = df[target].astype(int)   # binary labels as int
    return X, y

def get_models(naive_variant="gaussian"):
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
            n_estimators=300, random_state=42, class_weight="balanced_subsample"
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            objective="binary:logistic", eval_metric="logloss"
        )
    }
    return models

def main(args):
    df = load_data(args.data)
    # Basic rubric checks (>=500 instances, >=12 features) — ensure before submission
    assert df.shape[0] >= 500, "Dataset must have >= 500 instances."
    assert df.drop(columns=[args.target]).shape[1] >= 12, "Dataset must have >= 12 features."
    # Train/Val split
    X, y = split_X_y(df, args.target, args.drop)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor(X_tr)
    models = get_models(naive_variant=args.naive)

    results = {}
    for name, model in models.items():
        clf = Pipeline([("preprocess", preprocessor), ("model", model)])
        clf.fit(X_tr, y_tr)
        # predict_proba for ROC/AUC
        try:
            y_val_proba = clf.predict_proba(X_val)[:, 1]
        except Exception:
            # Fallback for models without predict_proba
            if hasattr(clf.named_steps["model"], "decision_function"):
                df_score = clf.named_steps["model"].decision_function(
                    preprocessor.transform(X_val)
                )
                df_min, df_max = df_score.min(), df_score.max()
                y_val_proba = (df_score - df_min) / (df_max - df_min + 1e-9)
            else:
                # Last resort: use binary predictions as pseudo-probabilities
                y_val_proba = clf.predict(X_val)

        metrics = compute_metrics(y_val, y_val_proba, threshold=args.threshold)
        results[name] = {m: metrics[m] for m in CLASSIFICATION_METRICS}
        results[name]["ConfusionMatrix"] = metrics["ConfusionMatrix"]
        results[name]["ClassificationReport"] = metrics["ClassificationReport"]

        # (Optional) Save model artifacts (commented for Streamlit free tier)
        # from joblib import dump
        # os.makedirs("model/artifacts", exist_ok=True)
        # dump(clf, f"model/artifacts/{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    # Save comparison table for README
    table = pd.DataFrame(results).T[CLASSIFICATION_METRICS]
    os.makedirs("model", exist_ok=True)
    table.to_csv("model/metrics_summary.csv", index=True)

    print("\n=== Validation Metrics Summary ===")
    print(table.round(4))

    with open("model/detailed_reports.txt", "w") as f:
        for name in results:
            f.write(f"\n\n## {name}\n")
            f.write(f"Confusion Matrix: {results[name]['ConfusionMatrix']}\n")
            f.write(f"{results[name]['ClassificationReport']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="train_data.csv", help="Path to labeled CSV (train)")
    parser.add_argument("--target", type=str, default="Depression", help="Target column name")
    parser.add_argument("--drop", nargs="*", default=DROP_DEFAULT, help="Columns to drop from features")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold for 0/1")
    parser.add_argument("--naive", choices=["gaussian", "multinomial"], default="gaussian", help="Naive Bayes type")
    args = parser.parse_args()
    main(args)
``
