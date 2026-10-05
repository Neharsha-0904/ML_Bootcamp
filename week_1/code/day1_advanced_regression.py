"""
ML Boot Camp - Day 1
Advanced Regression: From Model to Optimization

Dataset:
    Concrete Compressive Strength
Expected file:
    concrete.csv

Expected target column:
    Concrete compressive strength

Run:
    python day1_advanced_regression.py

The script demonstrates:
1. Data inspection
2. Train/test split
3. Baseline Linear Regression
4. OLS using the normal-equation linear system
5. Manual MSE
6. Gradient Descent from scratch
7. Learning-rate comparison
8. Feature scaling
9. Ridge Regression
10. Lasso Regression
11. Regularization strength / coefficient paths
12. Feature interactions
13. Residual analysis
14. Final comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_FILE = "concrete.csv"
TARGET = "Concrete compressive strength"
RANDOM_STATE = 42


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2


def print_metrics(name, y_true, y_pred):
    mae, mse, rmse, r2 = regression_metrics(y_true, y_pred)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Batch Gradient Descent for Mean Squared Error.

    X should already contain a bias/intercept column.
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    loss_history = []

    for _ in range(iterations):
        predictions = X @ beta
        error = predictions - y

        loss = np.mean(error ** 2)
        loss_history.append(loss)

        gradient = (2 / n_samples) * (X.T @ error)
        beta = beta - learning_rate * gradient

    return beta, loss_history


def add_bias(X):
    return np.c_[np.ones(X.shape[0]), X]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("DAY 1 - ADVANCED REGRESSION")
print("=" * 70)

print(f"\nLoading dataset: {DATA_FILE}")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"\nCould not find '{DATA_FILE}'. "
        "Place the CSV in the same folder as this script."
    )

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nColumns:")
for column in df.columns:
    print(" -", column)


# ============================================================
# 4. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("DATA INSPECTION")
print("=" * 70)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDescriptive statistics:")
print(df.describe().T)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

if TARGET not in df.columns:
    raise ValueError(
        f"\nTarget column '{TARGET}' was not found.\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. BASELINE LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("BASELINE LINEAR REGRESSION")
print("=" * 70)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_test)

print_metrics("Linear Regression", y_test, y_pred)

print("\nIntercept:")
print(linear_model.intercept_)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

print("\nLearned coefficients:")
print(coefficients.sort_values("Coefficient", key=np.abs, ascending=False).to_string(index=False))


# ============================================================
# 8. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Strength")
plt.ylabel("Predicted Strength")
plt.title("Day 1 - Actual vs Predicted")
plt.tight_layout()
plt.show()


# ============================================================
# 9. MATRIX FORMULATION
# ============================================================

print("\n" + "=" * 70)
print("MATRIX FORMULATION")
print("=" * 70)

X_train_np = X_train.to_numpy(dtype=float)
y_train_np = y_train.to_numpy(dtype=float)

X_bias = add_bias(X_train_np)

print("\nX shape:", X_train_np.shape)
print("X with bias shape:", X_bias.shape)
print("y shape:", y_train_np.shape)

print("\nModel in matrix form:")
print("    y_hat = X beta")


# ============================================================
# 10. OLS / NORMAL EQUATION
# ============================================================

print("\n" + "=" * 70)
print("ORDINARY LEAST SQUARES")
print("=" * 70)

print("\nNormal equation:")
print("    beta = (X^T X)^(-1) X^T y")

print("\nIn practice, we solve the linear system instead of explicitly")
print("forming the inverse.")

A = X_bias.T @ X_bias
b = X_bias.T @ y_train_np

beta_ols = np.linalg.solve(A, b)

print("\nOLS coefficients:")
print(beta_ols)

print("\nCompare with sklearn:")
print("Our intercept :", beta_ols[0])
print("Sklearn       :", linear_model.intercept_)

print("\nMaximum coefficient difference:",
      np.max(np.abs(beta_ols[1:] - linear_model.coef_)))


# ============================================================
# 11. MANUAL MSE
# ============================================================

print("\n" + "=" * 70)
print("MANUAL MSE")
print("=" * 70)

manual_train_predictions = X_bias @ beta_ols

manual_mse = np.mean(
    (y_train_np - manual_train_predictions) ** 2
)

sklearn_train_predictions = linear_model.predict(X_train)

sklearn_train_mse = mean_squared_error(
    y_train,
    sklearn_train_predictions
)

print("Manual MSE :", manual_mse)
print("Sklearn MSE:", sklearn_train_mse)


# ============================================================
# 12. FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE SCALING")
print("=" * 70)

print("\nOriginal feature ranges:")
print(X_train.describe().T[["min", "max"]])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_df = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

print("\nScaled feature summary:")
print(scaled_df.describe().T[["mean", "std", "min", "max"]])


# ============================================================
# 13. GRADIENT DESCENT FROM SCRATCH
# ============================================================

print("\n" + "=" * 70)
print("GRADIENT DESCENT FROM SCRATCH")
print("=" * 70)

X_gd = add_bias(X_train_scaled)
y_gd = y_train.to_numpy(dtype=float)

beta_gd, loss_history = gradient_descent(
    X_gd,
    y_gd,
    learning_rate=0.01,
    iterations=1000
)

print("\nLearned parameters:")
print(beta_gd)

plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Gradient Descent Convergence")
plt.tight_layout()
plt.show()

X_test_gd = add_bias(X_test_scaled)
gd_predictions = X_test_gd @ beta_gd

print_metrics(
    "Gradient Descent",
    y_test,
    gd_predictions
)


# ============================================================
# 14. LEARNING RATE EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("LEARNING RATE EXPERIMENT")
print("=" * 70)

learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0
]

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, losses = gradient_descent(
        X_gd,
        y_gd,
        learning_rate=lr,
        iterations=200
    )

    # Avoid plotting overflow values on a linear/log plot.
    losses = np.asarray(losses)

    if np.all(np.isfinite(losses)):
        plt.plot(
            losses,
            label=f"lr={lr}"
        )

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE (log scale)")
plt.title("Effect of Learning Rate")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 15. RIDGE REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("RIDGE REGRESSION")
print("=" * 70)

ridge = Ridge(alpha=10)
ridge.fit(X_train_scaled, y_train)

ridge_pred = ridge.predict(X_test_scaled)

print_metrics(
    "Ridge Regression",
    y_test,
    ridge_pred
)


# ============================================================
# 16. LASSO REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LASSO REGRESSION")
print("=" * 70)

lasso = Lasso(
    alpha=0.1,
    max_iter=10000
)

lasso.fit(X_train_scaled, y_train)

lasso_pred = lasso.predict(X_test_scaled)

print_metrics(
    "Lasso Regression",
    y_test,
    lasso_pred
)

print("\nNumber of zero Lasso coefficients:",
      np.sum(lasso.coef_ == 0))


# ============================================================
# 17. COEFFICIENT COMPARISON
# ============================================================

coefficient_comparison = pd.DataFrame({
    "Feature": X.columns,
    "Linear Regression": linear_model.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_
})

print("\nCoefficient comparison:")
print(coefficient_comparison.to_string(index=False))

coefficient_comparison.set_index("Feature").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.axhline(0, linestyle="--")
plt.ylabel("Coefficient")
plt.title("Linear Regression vs Ridge vs Lasso")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 18. RIDGE COEFFICIENT PATH
# ============================================================

print("\n" + "=" * 70)
print("RIDGE REGULARIZATION PATH")
print("=" * 70)

alphas = np.logspace(-3, 3, 30)
ridge_coefs = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    ridge_coefs.append(model.coef_)

ridge_coefs = np.array(ridge_coefs)

plt.figure(figsize=(10, 6))

for i, feature in enumerate(X.columns):
    plt.plot(
        alphas,
        ridge_coefs[:, i],
        label=feature
    )

plt.xscale("log")
plt.xlabel("Regularization strength α")
plt.ylabel("Coefficient")
plt.title("Ridge Coefficient Paths")
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()


# ============================================================
# 19. FEATURE INTERACTIONS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE INTERACTIONS")
print("=" * 70)

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("Original feature count:", X_train_scaled.shape[1])
print("Expanded feature count:", X_train_poly.shape[1])

interaction_model = Ridge(alpha=10)

interaction_model.fit(
    X_train_poly,
    y_train
)

interaction_pred = interaction_model.predict(
    X_test_poly
)

print_metrics(
    "Ridge + Degree-2 Feature Expansion",
    y_test,
    interaction_pred
)


# ============================================================
# 20. RESIDUAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Strength")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()


# ============================================================
# 21. FINAL MODEL COMPARISON
# ============================================================

models = {
    "Linear Regression": y_pred,
    "Gradient Descent": gd_predictions,
    "Ridge": ridge_pred,
    "Lasso": lasso_pred,
    "Ridge + Interactions": interaction_pred
}

rows = []

for name, predictions in models.items():
    mae, mse, rmse, r2 = regression_metrics(
        y_test,
        predictions
    )

    rows.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

results = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)
print(results.sort_values("RMSE").to_string(index=False))


# ============================================================
# 22. DAY 1 TAKEAWAY
# ============================================================

print("\n" + "=" * 70)
print("DAY 1 TAKEAWAY")
print("=" * 70)

print("""
Data
  ↓
Features + Target
  ↓
Linear Regression
  ↓
Prediction
  ↓
Loss / MSE
  ↓
OLS / Normal Equation
  ↓
Gradient Descent
  ↓
Feature Scaling
  ↓
Multicollinearity
  ↓
Ridge / Lasso
  ↓
Feature Interactions
  ↓
Validation + Diagnostics
  ↓
Generalization

DAY 2:
We will open the optimizer and study exactly how it moves
through a loss landscape, why it can fail, and how Momentum,
AdaGrad, RMSProp and Adam improve optimization.
""")
