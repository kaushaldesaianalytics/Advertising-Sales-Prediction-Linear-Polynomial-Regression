# Advertising Spend and Sales Forecasting
## Complete Regression Modeling Pipeline

Which advertising channels drive sales, and how much? This project builds a full regression pipeline on the Advertising dataset, progressing from a linear baseline through polynomial feature expansion and into regularized models (Ridge, Lasso, and ElasticNet) with cross-validation, GridSearchCV, and a side-by-side comparison of all approaches.

---

## Overview

This project consolidates the complete regression modeling workflow into a single end-to-end notebook. It is structured to demonstrate how model complexity, feature engineering, and regularization interact, and how to make principled decisions at each stage using train/test RMSE, cross-validation, and residual diagnostics.

The dataset is intentionally simple (200 observations, 3 features) so that every modeling decision is transparent and the results are fully interpretable. The techniques demonstrated here (degree selection, leak-free scaling, regularized regression, hyperparameter tuning) transfer directly to larger, more complex datasets.

---

## Dataset

The [Advertising dataset](https://www.kaggle.com/datasets/ashydv/advertising-dataset) contains 200 observations across four columns, all measured in thousands of dollars or units.

| Feature | Description |
|---|---|
| TV | Advertising spend on television |
| radio | Advertising spend on radio |
| newspaper | Advertising spend on newspaper |
| sales | Product sales (target variable) |

---

## Workflow

### Part 1: Exploratory Data Analysis
Scatter plots of each spend channel against sales confirm TV and radio carry the strongest linear signal. Newspaper spend shows a weaker relationship. This shapes feature engineering decisions downstream.

### Part 2: Linear Regression Baseline
A multiple linear regression model on raw features establishes benchmark metrics: MAE, RMSE, and R². Residual diagnostics (scatter plot, distribution plot, and Q-Q plot) validate that model assumptions hold before moving to more complex approaches.

### Part 3: Polynomial Feature Expansion
`PolynomialFeatures` expands the 3-column input into interaction and higher-order terms. At degree 2 this produces 9 features; at degree 3 it produces 19. A degree sweep from 1 to 9 tracks train and test RMSE to identify the optimal degree before overfitting. Degree 3 is selected. The polynomial model is evaluated with the same diagnostic suite as the linear baseline.

### Part 4: Feature Scaling
Regularized models penalize large coefficients. Without scaling, features on larger numeric ranges absorb larger penalties regardless of their true predictive value. StandardScaler normalizes all polynomial features to zero mean and unit variance. The scaler is fit only on the training set to prevent data leakage.

### Part 5: Ridge Regression
Ridge adds an L2 penalty to the loss function, shrinking all coefficients toward zero without zeroing any. Two approaches are compared: a fixed alpha=10 baseline and RidgeCV, which selects the optimal alpha via cross-validation across a log-spaced grid.

### Part 6: Lasso Regression
Lasso adds an L1 penalty, which drives low-signal coefficients exactly to zero. This performs implicit feature selection, which is useful when many of the polynomial-expanded features carry little independent signal. LassoCV identifies the optimal alpha from a regularization path.

### Part 7: ElasticNet
ElasticNet combines L1 and L2 penalties, controlled by alpha (regularization strength) and l1_ratio (the L1/L2 balance). Two tuning strategies are compared: ElasticNetCV sweeps l1_ratio with built-in cross-validation, and GridSearchCV jointly optimizes both alpha and l1_ratio across a full parameter grid.

### Part 8: Model Comparison
All seven models are compared side-by-side in a results table and bar chart visualization, ranked by MAE and RMSE. This provides a clear view of where each modeling choice improved or worsened performance.

### Part 9: Model Persistence and Prediction
The final degree-3 polynomial model is saved to disk using `joblib`. Both the fitted `PolynomialFeatures` converter and the trained model are persisted, and a demonstration shows how to reload and predict on a new campaign budget without re-fitting.

---

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~1.51 | ~1.95 | ~0.90 |
| Polynomial deg=3 | ~0.49 | ~0.66 | ~0.98 |
| Ridge (alpha=10) | varies | varies | varies |
| RidgeCV | varies | varies | varies |
| LassoCV | varies | varies | varies |
| ElasticNetCV | varies | varies | varies |
| ElasticNet GridCV | varies | varies | varies |

Polynomial expansion delivers the largest single performance gain. Regularized models on polynomial features add stability and reduce overfitting risk, particularly for higher-degree expansions.

---

## Key Concepts

**Why Polynomial Outperforms Linear Here:** TV and radio spend interact multiplicatively. Combined investment yields disproportionate returns. Linear regression cannot capture this without an explicit interaction term; polynomial expansion creates it automatically.

**Scaling Before Regularization:** Without scaling, a feature measured in hundreds (TV spend) would receive a smaller regularization penalty than one measured in single digits (newspaper), regardless of predictive value. Scaling ensures the penalty is applied fairly across all features.

**L1 vs. L2 Penalties:** L1 (Lasso) produces sparse models by zeroing out low-signal coefficients. L2 (Ridge) shrinks all coefficients smoothly. ElasticNet provides a tunable blend. In a polynomial-expanded feature space with many correlated terms, some sparsity is generally beneficial.

**GridSearchCV inside a Pipeline:** When scaling is part of the pipeline, GridSearchCV refits the scaler on each training fold independently. Without a pipeline, the scaler would be fit on the full training set once, and test-fold data would influence the scaling. This is a subtle but important form of data leakage.

**Model Persistence with joblib:** Saving the feature transformer alongside the model is essential. A loaded model expects inputs in the same format it was trained on. If only the model is saved, raw inputs will produce incorrect predictions.

---

## Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn, SciPy
- scikit-learn (LinearRegression, PolynomialFeatures, StandardScaler, Ridge, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, GridSearchCV, Pipeline, metrics)
- joblib

---

## File Structure

```
advertising-regression-complete/
├── advertising_regression_complete.ipynb   # Main project notebook
├── Advertising.csv                         # Dataset
├── final_poly_model.joblib                 # Saved trained model
├── final_poly_converter.joblib             # Saved polynomial feature transformer
└── README.md
```

---

## How to Run

1. Clone the repository
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scipy scikit-learn joblib`
3. Open `advertising_regression_complete.ipynb` in Jupyter and run all cells
