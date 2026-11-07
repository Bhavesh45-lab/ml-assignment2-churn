"""
Telco Customer Churn - Model Evaluation Dashboard
ML Assignment 2, M.Tech (AIML), BITS Pilani WILP
Student: Bhavesh Jadhav (BITS ID: 2026AJ05850)

Upload the held-out test CSV, pick one of the five pre-trained classifiers and see
its metrics, confusion matrix and classification report. Models are only loaded
from model/*.joblib - nothing is retrained on the uploaded data.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
TARGET = "Churn"
FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen", "Partner",
    "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
STUDENT_NAME = "Bhavesh Jadhav"
BITS_ID = "2026AJ05850"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Churn Model Lab", page_icon="📉", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_best_params():
    p = MODEL_DIR / "best_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


def prepare(df):
    """Apply the same light cleaning as training. Returns X, y (0/1)."""
    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    if pd.api.types.is_numeric_dtype(df["SeniorCitizen"]):
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = df[TARGET]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().map({"No": 0, "Yes": 1})
    return df[FEATURES], y.astype(int)


def predict_labels(model, X, y_prob, threshold):
    """At the default threshold use model.predict() (same as training/README);
    otherwise apply the user-chosen threshold to the predicted probability.
    (For SVC, predict() uses the decision function, which can differ slightly
    from predict_proba >= 0.5 because of Platt scaling.)"""
    if abs(threshold - 0.5) < 1e-9:
        return model.predict(X)
    return (y_prob >= threshold).astype(int)


def compute_metrics(y, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📉 Churn Model Lab")
st.sidebar.markdown(f"**{STUDENT_NAME}**  \nBITS ID: `{BITS_ID}`")
st.sidebar.caption("ML Assignment 2 · M.Tech AIML · BITS WILP")

uploaded = st.sidebar.file_uploader("1 · Upload test data (CSV)", type=["csv"])
sample = BASE / "test_data.csv"
if sample.exists():
    st.sidebar.download_button("⬇ Download sample test_data.csv", sample.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

model_name = st.sidebar.selectbox("2 · Choose a model", list(MODEL_FILES))
threshold = st.sidebar.slider("Decision threshold for 'Churn = Yes'", 0.05, 0.95, 0.50, 0.05,
                              help="Default 0.50. Lower it to catch more churners (higher recall).")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** IBM Telco Customer Churn (Kaggle)  \n"
    "**Task:** binary classification — will the customer leave?  \n"
    "**Models:** pre-trained sklearn pipelines loaded from `model/`."
)

# ---------------------------------------------------------------- main
st.title("Telco Customer Churn — Classifier Evaluation")
st.caption(f"Submitted by **{STUDENT_NAME}** · BITS ID **{BITS_ID}** · M.Tech (AIML) · Machine Learning — Assignment 2")
st.write("Upload the held-out test set, pick a classifier and compare how well each "
         "pre-trained model identifies customers who are about to churn.")

if uploaded is None:
    st.info("👈 Upload a test CSV from the sidebar to begin. It must contain the 19 input "
            "features and the `Churn` column. Use the download button to get the sample file.")
    with st.expander("Expected columns"):
        st.code(", ".join(FEATURES + [TARGET]))
    st.stop()

raw = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in raw.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

X, y = prepare(raw)
models = load_models()

c1, c2, c3 = st.columns(3)
c1.metric("Rows in test file", f"{len(X):,}")
c2.metric("Actual churners", f"{int(y.sum()):,}")
c3.metric("Churn rate", f"{y.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head(20), width="stretch")

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all 5 models"])

# ---------------------------------------------------------------- single model
with tab_single:
    model = models[model_name]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = predict_labels(model, X, y_prob, threshold)
    m = compute_metrics(y, y_pred, y_prob)

    st.subheader(f"Evaluation metrics — {model_name}")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")
    params = load_best_params().get(model_name)
    if params:
        st.caption(f"Tuned hyper-parameters (GridSearchCV, 5-fold, F1): `{params}` · "
                   f"threshold = {threshold:.2f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Stay (0)", "Churn (1)"], yticklabels=["Stay (0)", "Churn (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        tn, fp, fn, tp = cm.ravel()
        st.caption(f"Caught {tp} of {tp + fn} churners; {fp} loyal customers wrongly flagged.")
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, y_prob)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)

    st.markdown("**Classification report**")
    rep = classification_report(y, y_pred, target_names=["No Churn", "Churn"],
                                output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.round(4), width="stretch")

    out = raw.copy()
    out["Predicted_Churn"] = np.where(y_pred == 1, "Yes", "No")
    out["Churn_Probability"] = y_prob.round(4)
    st.download_button("⬇ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv",
                       mime="text/csv")

# ---------------------------------------------------------------- comparison
with tab_compare:
    st.subheader("All models on the uploaded test data")
    rows = {}
    for name, mdl in models.items():
        p = mdl.predict_proba(X)[:, 1]
        rows[name] = compute_metrics(y, predict_labels(mdl, X, p, threshold), p)
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    best = table["MCC"].idxmax()
    st.success(f"Best model by MCC on this file: **{best}** (MCC = {table.loc[best, 'MCC']:.4f})")

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    order = table[metric].sort_values()
    ax.barh(order.index, order.values, color=["#1f77b4" if i != best else "#2ca02c" for i in order.index])
    for i, v in enumerate(order.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, min(1.0, order.max() + 0.1))
    ax.set_xlabel(metric)
    st.pyplot(fig)
