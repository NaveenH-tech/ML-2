# ml_core.py

import os
import pickle
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -------------------------------------------------
# Dense transformer (needed for Naive Bayes)
# -------------------------------------------------
class DenseTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


# -------------------------------------------------
# Standard preprocessor (for most models)
# -------------------------------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


# -------------------------------------------------
# Preprocessor for Multinomial Naive Bayes
# No scaling. No negative values.
# -------------------------------------------------
def build_preprocessor_for_multinomial_nb(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


# -------------------------------------------------
# Model definitions
# -------------------------------------------------
def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes Gaussian": GaussianNB(),
        "Naive Bayes Multinomial": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )

    return models


# -------------------------------------------------
# Train and save all models
# -------------------------------------------------
def train_and_save_all_models(X_train, y_train, output_path):
    print(" Building preprocessor")
    models = get_models()
    trained_models = {}

    for model_name, model in models.items():
        print(" Training model:", model_name)

        # Multinomial NB (no scaling, dense, non-negative)
        if model_name == "Naive Bayes Multinomial":
            preprocessor = build_preprocessor_for_multinomial_nb(X_train)
            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("dense", DenseTransformer()),
                ("model", model)
            ])

        # Gaussian NB (needs dense)
        elif model_name == "Naive Bayes Gaussian":
            preprocessor = build_preprocessor(X_train)
            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("dense", DenseTransformer()),
                ("model", model)
            ])

        # All other models
        else:
            preprocessor = build_preprocessor(X_train)
            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("model", model)
            ])

        pipeline.fit(X_train, y_train)
        trained_models[model_name] = pipeline

        print(" Completed:", model_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(trained_models, f)

    print(" All models trained")
    print(" Models saved at:", output_path)
