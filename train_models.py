"""
ML Assignment 2 - Telco Customer Churn Prediction
M.Tech (AIML), BITS Pilani WILP

Trains 5 classifiers on the IBM Telco Customer Churn dataset, evaluates them on a
held-out stratified test split and saves:
    model/<name>.joblib      - full sklearn Pipeline (preprocessing + classifier)
    model/metrics.json       - test-set metrics for every model
    model/best_params.json   - hyper-parameters chosen by GridSearchCV
    test_data.csv            - raw held-out test rows (features + Churn column)

Run:  python train_models.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
DATA_PATH = "data/Telco-Customer-Churn.csv"
MODEL_DIR = "model"
TARGET = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges is stored as text; 11 new customers (tenure = 0) have a blank value.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    return df


def make_preprocessor():
    # Dense output so that GaussianNB can also consume it.
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
    ])


# Model name -> (estimator, small hyper-parameter grid)
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [4, 5, 6, 8, 10],
         "clf__min_samples_leaf": [10, 20, 50],
         "clf__criterion": ["gini", "entropy"]},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 31, 41],
         "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
    ),
    "Support Vector Machine": (
        SVC(probability=True, random_state=SEED),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["linear", "rbf"]},
    ),
}


def file_name(model_name):
    return model_name.lower().replace(" ", "_") + ".joblib"


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_and_clean(DATA_PATH)
    print(f"Dataset shape: {df.shape} | churn rate: {df[TARGET].mean():.3f}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Save the raw test split - this is the file uploaded into the Streamlit app.
    test_df = X_test.copy()
    test_df[TARGET] = y_test.map({0: "No", 1: "Yes"})
    test_df.to_csv("test_data.csv", index=False)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows -> test_data.csv")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_metrics, best_params = {}, {}

    for name, (estimator, grid) in MODELS.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", estimator)])
        # Tune on training data only, using F1 (churn class is the minority).
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics = evaluate(best, X_test, y_test)
        all_metrics[name] = {k: round(float(v), 4) for k, v in metrics.items()}
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        joblib.dump(best, os.path.join(MODEL_DIR, file_name(name)))
        print(f"{name:<24} best={best_params[name]}  " +
              "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open(os.path.join(MODEL_DIR, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    table = pd.DataFrame(all_metrics).T
    print("\nComparison table (test set):")
    print(table.to_markdown(floatfmt=".4f"))


if __name__ == "__main__":
    main()
