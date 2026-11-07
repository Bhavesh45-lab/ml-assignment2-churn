# Telco Customer Churn Prediction — ML Assignment 2

**M.Tech (AIML) · Machine Learning · BITS Pilani WILP**

**Student:** Bhavesh Jadhav  |  **BITS ID:** 2026AJ05850

- **GitHub repository:** `<paste your GitHub repo link here>`
- **Live Streamlit app:** `<paste your Streamlit app link here>`

---

## a. Problem statement

A telecom operator loses revenue every time a subscriber cancels their service ("churns"). Keeping an existing customer is much cheaper than acquiring a new one, so the business wants to flag customers who are likely to leave early enough to offer them a retention deal.

The task is a **binary classification problem**: using a customer's account details, subscribed services and billing information, predict whether the customer will churn (`Churn = Yes`) or stay (`Churn = No`). Five classical classifiers — Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine — are trained on the same data, compared on six evaluation metrics, and served through an interactive Streamlit app that evaluates the saved models on uploaded test data.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample dataset) |
| Source | Kaggle — https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Instances | 7,043 customers |
| Input features | 19 (after dropping the `customerID` identifier) |
| Target | `Churn` — Yes (1,869 / 26.5%) vs No (5,174 / 73.5%); moderately imbalanced |
| Missing values | 11 blank `TotalCharges` values, all for brand-new customers with `tenure = 0` → set to 0 |

**Features**

| Group | Features | Type |
|---|---|---|
| Account | `tenure` (months), `Contract` (month-to-month / one year / two year), `PaperlessBilling`, `PaymentMethod` (4 types) | numeric + categorical |
| Charges | `MonthlyCharges`, `TotalCharges` | numeric |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | categorical |
| Services | `PhoneService`, `MultipleLines`, `InternetService` (DSL / Fiber optic / No), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | categorical |

**Pre-processing** (inside an sklearn `Pipeline`, so it is saved with each model and applied automatically in the app):

- `TotalCharges` converted from text to numeric; `SeniorCitizen` 0/1 mapped to No/Yes.
- 3 numeric features → `StandardScaler` (needed by KNN, SVM and Logistic Regression).
- 16 categorical features → `OneHotEncoder` (`handle_unknown="ignore"`), giving 46 model inputs.
- Stratified 80 / 20 train–test split (`random_state = 42`): **5,634 training rows, 1,409 test rows**. The test rows are saved, unprocessed, as `test_data.csv`.
- Hyper-parameters were tuned with `GridSearchCV` (5-fold stratified CV, scoring = F1) **on the training set only**; the test set was used once, for the final numbers below.

## c. GitHub repository

`<paste your GitHub repo link here>`

```
ml-assignment2-churn/
├── app.py                     # Streamlit app
├── train_models.py            # preprocessing, tuning, training, evaluation, saving
├── requirements.txt
├── README.md
├── test_data.csv              # 1,409 held-out test rows (19 features + Churn)
├── data/
│   └── Telco-Customer-Churn.csv   # full original dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    ├── metrics.json           # test-set metrics of all models
    └── best_params.json       # tuned hyper-parameters
```

## d. Models used

| Model | Tuned hyper-parameters |
|---|---|
| Logistic Regression | C = 10 |
| Decision Tree | criterion = entropy, max_depth = 5, min_samples_leaf = 50 |
| KNN | n_neighbors = 41, weights = uniform |
| Naive Bayes (Gaussian) | var_smoothing = 0.01 |
| Support Vector Machine | kernel = linear, C = 1 (probability = True for AUC) |

### Comparison table (test set, 1,409 customers, threshold = 0.5)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8041** | **0.8412** | **0.6541** | 0.5561 | **0.6012** | **0.4752** |
| Decision Tree | 0.7956 | 0.8304 | 0.6405 | 0.5241 | 0.5765 | 0.4474 |
| KNN | 0.7835 | 0.8307 | 0.5925 | 0.5909 | 0.5917 | 0.4444 |
| Naive Bayes | 0.6955 | 0.8079 | 0.4597 | **0.8396** | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7878 | 0.8246 | 0.6176 | 0.5267 | 0.5685 | 0.4314 |

Precision, Recall and F1 are for the positive class (`Churn = Yes`). Because only 26.5% of customers churn, a model that always predicts "No" would already reach 73.5% accuracy, so **MCC, F1 and AUC are more informative than accuracy** for this dataset.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best on five of six metrics (Accuracy 0.804, AUC 0.841, F1 0.601, MCC 0.475). After one-hot encoding, churn risk is close to additive in the features — fibre-optic internet and a month-to-month contract push the log-odds up, while longer tenure and a two-year contract pull them down — so a linear model fits well and generalises without overfitting. It also gives well-calibrated probabilities and interpretable coefficients, which is useful to the business. Its weakness is recall (0.556): at the default 0.5 threshold it misses about 44% of churners. |
| Decision Tree | An unrestricted tree overfits badly here (99.8% train accuracy vs 71.8% test, MCC 0.28), so grid search picked a shallow tree (depth 5, ≥ 50 samples per leaf). That tree is competitive (Accuracy 0.796, MCC 0.447) and easy to explain as rules (root split: month-to-month contract, then fibre-optic internet or monthly charges, then tenure), but its piece-wise constant probabilities give a slightly lower AUC (0.830) and it has the second-lowest recall (0.524). |
| KNN | Needed a large neighbourhood (k = 41) to smooth out noise in the 46-dimensional one-hot space. It has the most balanced precision/recall (0.593 / 0.591) and a good AUC (0.831), but overall MCC (0.444) is below Logistic Regression. Distances are dominated by the many binary dummies, and KNN must keep all 5,634 training rows, so it is the largest model file and the slowest to predict. |
| Naive Bayes | Behaves very differently from the others: highest recall (0.840 — it catches 314 of 374 churners) but lowest precision (0.460) and accuracy (0.696). The independence assumption is clearly violated (e.g. `TotalCharges ≈ tenure × MonthlyCharges`, and the "No internet service" dummies are perfectly correlated), so it double-counts evidence and pushes probabilities to the extremes, over-predicting churn. It is still useful if missing a churner is far costlier than a wasted retention offer. |
| Support Vector Machine | Grid search preferred a linear kernel over RBF, again showing that the classes are roughly linearly separable. However, the hinge loss optimises the margin rather than the likelihood, and its Platt-scaled probabilities were slightly worse: all metrics are a little below Logistic Regression (MCC 0.431, AUC 0.825, recall 0.527). It is also the slowest model to train because of `probability=True` (internal 5-fold calibration). |
| **Overall winner for this dataset** | **Logistic Regression** — highest Accuracy, AUC, Precision, F1 and MCC, plus it is fast, small and interpretable. If the business goal is to catch as many churners as possible, **Naive Bayes** (recall 0.84) or Logistic Regression with a lower decision threshold (adjustable in the app) would be preferred. |

## e. Streamlit app features

1. **CSV upload** of test data (sidebar) with column validation, plus a button to download the sample `test_data.csv`.
2. **Model selection dropdown** for the five saved classifiers.
3. **Evaluation metrics** — Accuracy, AUC, Precision, Recall, F1 and MCC shown as metric cards.
4. **Confusion matrix** heat-map, **ROC curve** and full **classification report**.
5. **Compare-all tab** — metrics of all five models on the uploaded file, best values highlighted, bar chart for any chosen metric.
6. Adjustable **decision threshold** slider and **download of predictions** with churn probabilities.

The app only loads the pre-trained pipelines from `model/` and calls `predict_proba`; it never retrains on uploaded data.

## f. How to run

```bash
pip install -r requirements.txt
python train_models.py        # optional: re-trains and regenerates model/ and test_data.csv
streamlit run app.py
```

Deployment: Streamlit Community Cloud → New app → this repository → branch `main` → `app.py`.
