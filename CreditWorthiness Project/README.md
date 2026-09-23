# Creditworthiness Risk Model

A binary classification project that predicts the probability of a borrower defaulting on credit, using a synthetically generated dataset of borrower financial attributes. The notebook compares a **Logistic Regression** baseline against an **XGBoost** challenger model.

## Overview

This project simulates a typical credit risk scoring workflow:

1. Generate a synthetic dataset of borrower financial features
2. Derive a latent risk score and use it to create a realistic, imbalanced binary target (`default`)
3. Engineer standard credit-risk features (DTI ratio, credit utilization)
4. Train and compare two classification models
5. Evaluate performance using metrics appropriate for imbalanced classification

## Dataset

The dataset is synthetically generated (`n = 5,000` samples, `random_state = 42`) with the following raw features:

| Feature | Description | Distribution |
|---|---|---|
| `monthly_income` | Borrower's monthly income | Normal(5000, 2000), clipped [1000, 20000] |
| `monthly_debt_payments` | Total monthly debt obligations | Normal(1500, 800), clipped [0, 10000] |
| `revolving_balance` | Outstanding revolving credit balance | Exponential(4000) |
| `total_credit_limit` | Total available credit limit | Normal(15000, 7000), clipped [2000, 50000] |
| `delinquencies_90_days` | Count of 90-day delinquencies | Poisson(0.2) |
| `account_tenure_months` | Age of credit account, in months | Uniform integer [6, 240] |
| `recent_inquiries` | Recent hard credit inquiries | Poisson(0.5) |

### Target Variable

A latent risk score is computed as a weighted combination of debt-to-income, credit utilization, delinquencies, recent inquiries, and account tenure. Borrowers in the top 15% of latent risk are labeled `default = 1`, producing a realistic **~85/15 class imbalance**, similar to real-world credit portfolios.

## Feature Engineering

Two derived features are added prior to modeling:

- **`dti_ratio`** — `monthly_debt_payments / monthly_income`
- **`credit_utilization`** — `revolving_balance / total_credit_limit`

## Methodology

- **Train/test split:** 80/20, stratified on the target to preserve class balance
- **Scaling:** Features are standardized with `StandardScaler` for Logistic Regression; XGBoost is trained on unscaled data since tree-based models don't require it
- **Class imbalance handling:**
  - Logistic Regression uses `class_weight='balanced'`
  - XGBoost uses `scale_pos_weight` computed from the training set's class ratio

## Models

| Model | Library | Notes |
|---|---|---|
| Logistic Regression | `scikit-learn` | Baseline, balanced class weights |
| XGBoost | `xgboost` | `n_estimators=100`, `max_depth=4`, `learning_rate=0.1` |

## Evaluation

Both models are evaluated on the held-out test set using:

- **Classification report** (precision, recall, F1 for both "Good Risk" and "Default" classes)
- **ROC-AUC score**

These metrics are chosen over raw accuracy because of the class imbalance in the target variable.

## Requirements

```
numpy
pandas
scikit-learn
xgboost
```

## Usage

1. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost
   ```
2. Open and run the notebook top to bottom:
   ```bash
   jupyter notebook Creditworthiness.ipynb
   ```
3. Review the printed classification reports and ROC-AUC scores for both models in the output cells.

## Project Structure

```
.
├── Creditworthiness.ipynb   # Main notebook: data generation, feature engineering, modeling, evaluation
└── README.md                 # Project documentation
```

## Notes & Caveats

- **Synthetic data:** All borrower records are artificially generated, not sourced from real credit data. Model performance figures reflect the synthetic generation process and should not be interpreted as production-grade credit risk estimates.
- **No hyperparameter tuning:** Model parameters are set to reasonable defaults; neither model is tuned via cross-validation or grid/random search.
- **Fixed random seed:** `random_state=42` is used throughout for reproducibility.

## Possible Extensions

- Hyperparameter tuning (e.g., `GridSearchCV`, `Optuna`)
- Feature importance / SHAP analysis for model interpretability
- Calibration analysis of predicted probabilities
- Cross-validation instead of a single train/test split
- Additional derived features (e.g., payment history trends, credit mix)
