"""
ML Boot Camp - Day 2
Optimization Under the Hood

This script is intentionally built around experiments.

It demonstrates:
1. Loss landscapes
2. Gradient direction
3. Gradient Descent from scratch
4. Optimization trajectories
5. Learning-rate failure
6. Feature scaling / conditioning
7. Batch vs SGD vs Mini-Batch
8. Momentum
9. AdaGrad
10. RMSProp
11. Adam
12. Regularized objectives
13. Ridge-style optimization
14. A tiny neural-network training bridge

No external dataset is required.
The optimizer experiments use synthetic functions and synthetic
regression data so that the optimization behaviour is easy to see.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# ============================================================
# 1. QUADRATIC LOSS LANDSCAPE
# ============================================================

print("=" * 70)
print("DAY 2 - OPTIMIZATION UNDER THE HOOD")
print("=" * 70)

print("""
Core loop:

Parameters
    ↓
Prediction / Objective
    ↓
Loss
    ↓
Gradient
    ↓
Optimizer
    ↓
Parameter update
    ↓
Repeat
""")


def quadratic(theta):
    """
    A simple convex 2D objective.

    f(x, y) = 0.5 * (x^2 + 10*y^2)

    The y direction is much steeper than the x direction.
    This creates an elongated valley and makes zig-zag behaviour visible.
    """
    x, y = theta
    return 0.5 * (x**2 + 10.0 * y**2)


def quadratic_gradient(theta):
    x, y = theta
    return np.array([x, 10.0 * y])


# ============================================================
# 2. LOSS LANDSCAPE VISUALIZATION
# ============================================================

x = np.linspace(-5, 5, 300)
y = np.linspace(-5, 5, 300)

X, Y = np.meshgrid(x, y)
Z = 0.5 * (X**2 + 10.0 * Y**2)

plt.figure(figsize=(9, 7))
contour = plt.contour(X, Y, Z, levels=30)
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel("Parameter θ₁")
plt.ylabel("Parameter θ₂")
plt.title("Convex Loss Landscape")
plt.scatter([0], [0], s=80, label="Global minimum")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 3. GENERIC GRADIENT DESCENT
# ============================================================

def gradient_descent(
    theta0,
    gradient_fn,
    learning_rate=0.05,
    iterations=100,
):
    theta = np.array(theta0, dtype=float)
    history = [theta.copy()]
    losses = [quadratic(theta)]

    for _ in range(iterations):
        gradient = gradient_fn(theta)
        theta = theta - learning_rate * gradient

        history.append(theta.copy())
        losses.append(quadratic(theta))

    return theta, np.array(history), np.array(losses)


theta0 = np.array([4.0, 4.0])

theta_final, trajectory, losses = gradient_descent(
    theta0,
    quadratic_gradient,
    learning_rate=0.05,
    iterations=100
)

print("\nGradient Descent")
print("Start:", theta0)
print("Final:", theta_final)
print("Final loss:", losses[-1])


# ============================================================
# 4. OPTIMIZATION TRAJECTORY
# ============================================================

plt.figure(figsize=(9, 7))
contour = plt.contour(X, Y, Z, levels=30)
plt.clabel(contour, inline=True, fontsize=8)

plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    marker="o",
    markersize=2,
    linewidth=1
)

plt.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    s=100,
    label="Start"
)

plt.scatter(
    0,
    0,
    s=100,
    label="Minimum"
)

plt.xlabel("Parameter θ₁")
plt.ylabel("Parameter θ₂")
plt.title("Gradient Descent Trajectory")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 5. LEARNING RATE EXPERIMENT
# ============================================================

learning_rates = [
    0.001,
    0.02,
    0.05,
    0.15,
    0.21
]

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, _, lr_losses = gradient_descent(
        theta0,
        quadratic_gradient,
        learning_rate=lr,
        iterations=80
    )

    if np.all(np.isfinite(lr_losses)):
        plt.plot(
            lr_losses,
            label=f"lr={lr}"
        )

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Learning Rate: Slow, Stable and Unstable Behaviour")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 6. OPTIMIZATION WITH MOMENTUM
# ============================================================

def momentum_optimizer(
    theta0,
    gradient_fn,
    learning_rate=0.05,
    momentum=0.9,
    iterations=100
):
    theta = np.array(theta0, dtype=float)
    velocity = np.zeros_like(theta)

    history = [theta.copy()]
    losses = [quadratic(theta)]

    for _ in range(iterations):
        gradient = gradient_fn(theta)

        velocity = momentum * velocity + learning_rate * gradient
        theta = theta - velocity

        history.append(theta.copy())
        losses.append(quadratic(theta))

    return theta, np.array(history), np.array(losses)


theta_m, trajectory_m, losses_m = momentum_optimizer(
    theta0,
    quadratic_gradient,
    learning_rate=0.03,
    momentum=0.9,
    iterations=100
)

print("\nMomentum final parameters:", theta_m)
print("Momentum final loss:", losses_m[-1])


# ============================================================
# 7. COMPARE GD AND MOMENTUM TRAJECTORIES
# ============================================================

_, trajectory_gd, losses_gd = gradient_descent(
    theta0,
    quadratic_gradient,
    learning_rate=0.05,
    iterations=100
)

plt.figure(figsize=(9, 7))

contour = plt.contour(X, Y, Z, levels=30)
plt.clabel(contour, inline=True, fontsize=8)

plt.plot(
    trajectory_gd[:, 0],
    trajectory_gd[:, 1],
    label="Gradient Descent"
)

plt.plot(
    trajectory_m[:, 0],
    trajectory_m[:, 1],
    label="Momentum"
)

plt.scatter(0, 0, s=100, label="Minimum")

plt.xlabel("Parameter θ₁")
plt.ylabel("Parameter θ₂")
plt.title("Gradient Descent vs Momentum")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. SYNTHETIC REGRESSION DATA
# ============================================================

n_samples = 2000
n_features = 10

X_data = np.random.randn(n_samples, n_features)

true_weights = np.array([
    3.0, -2.0, 1.5, 0.5, -1.0,
    2.0, 0.0, 0.8, -0.5, 1.2
])

noise = np.random.randn(n_samples) * 0.5

y_data = X_data @ true_weights + noise

# Standardize target only for numerical convenience.
# Features already have approximately standard-normal scale.
y_data = (y_data - y_data.mean()) / y_data.std()

X_train = X_data[:1600]
y_train = y_data[:1600]

X_test = X_data[1600:]
y_test = y_data[1600:]


# ============================================================
# 9. BATCH GD / SGD / MINI-BATCH
# ============================================================

def mse_gradient(X, y, theta):
    predictions = X @ theta
    error = predictions - y
    return (2 / len(X)) * X.T @ error


def batch_gradient_descent(
    X,
    y,
    learning_rate=0.05,
    epochs=50
):
    theta = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        gradient = mse_gradient(X, y, theta)
        theta -= learning_rate * gradient

        predictions = X @ theta
        losses.append(np.mean((predictions - y) ** 2))

    return theta, losses


def stochastic_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=20
):
    theta = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        indices = np.random.permutation(len(X))

        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]

            gradient = mse_gradient(xi, yi, theta)
            theta -= learning_rate * gradient

        predictions = X @ theta
        losses.append(np.mean((predictions - y) ** 2))

    return theta, losses


def mini_batch_gradient_descent(
    X,
    y,
    learning_rate=0.05,
    batch_size=32,
    epochs=50
):
    theta = np.zeros(X.shape[1])
    losses = []

    for _ in range(epochs):
        indices = np.random.permutation(len(X))

        for start in range(0, len(X), batch_size):
            batch_indices = indices[start:start + batch_size]

            xb = X[batch_indices]
            yb = y[batch_indices]

            gradient = mse_gradient(xb, yb, theta)
            theta -= learning_rate * gradient

        predictions = X @ theta
        losses.append(np.mean((predictions - y) ** 2))

    return theta, losses


theta_batch, loss_batch = batch_gradient_descent(
    X_train,
    y_train
)

theta_sgd, loss_sgd = stochastic_gradient_descent(
    X_train,
    y_train
)

theta_mini, loss_mini = mini_batch_gradient_descent(
    X_train,
    y_train
)

plt.figure(figsize=(10, 6))

plt.plot(loss_batch, label="Batch GD")
plt.plot(loss_sgd, label="SGD")
plt.plot(loss_mini, label="Mini-Batch GD")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Batch vs SGD vs Mini-Batch")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. ADAGRAD
# ============================================================

def adagrad(
    theta0,
    gradient_fn,
    learning_rate=0.2,
    epsilon=1e-8,
    iterations=100
):
    theta = np.array(theta0, dtype=float)
    accumulated = np.zeros_like(theta)

    history = [theta.copy()]
    losses = [quadratic(theta)]

    for _ in range(iterations):
        gradient = gradient_fn(theta)

        accumulated += gradient ** 2

        adjusted_gradient = (
            gradient /
            (np.sqrt(accumulated) + epsilon)
        )

        theta -= learning_rate * adjusted_gradient

        history.append(theta.copy())
        losses.append(quadratic(theta))

    return theta, np.array(history), np.array(losses)


theta_a, trajectory_a, losses_a = adagrad(
    theta0,
    quadratic_gradient,
    learning_rate=0.3,
    iterations=100
)


# ============================================================
# 11. RMSPROP
# ============================================================

def rmsprop(
    theta0,
    gradient_fn,
    learning_rate=0.1,
    decay=0.9,
    epsilon=1e-8,
    iterations=100
):
    theta = np.array(theta0, dtype=float)
    squared_avg = np.zeros_like(theta)

    history = [theta.copy()]
    losses = [quadratic(theta)]

    for _ in range(iterations):
        gradient = gradient_fn(theta)

        squared_avg = (
            decay * squared_avg
            + (1 - decay) * gradient ** 2
        )

        theta -= (
            learning_rate
            * gradient
            / (np.sqrt(squared_avg) + epsilon)
        )

        history.append(theta.copy())
        losses.append(quadratic(theta))

    return theta, np.array(history), np.array(losses)


theta_r, trajectory_r, losses_r = rmsprop(
    theta0,
    quadratic_gradient,
    learning_rate=0.1,
    iterations=100
)


# ============================================================
# 12. ADAM
# ============================================================

def adam(
    theta0,
    gradient_fn,
    learning_rate=0.1,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    iterations=100
):
    theta = np.array(theta0, dtype=float)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    history = [theta.copy()]
    losses = [quadratic(theta)]

    for t in range(1, iterations + 1):
        gradient = gradient_fn(theta)

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        theta -= (
            learning_rate
            * m_hat
            / (np.sqrt(v_hat) + epsilon)
        )

        history.append(theta.copy())
        losses.append(quadratic(theta))

    return theta, np.array(history), np.array(losses)


theta_adam, trajectory_adam, losses_adam = adam(
    theta0,
    quadratic_gradient,
    learning_rate=0.1,
    iterations=100
)


# ============================================================
# 13. OPTIMIZER LOSS COMPARISON
# ============================================================

_, _, gd_loss = gradient_descent(
    theta0,
    quadratic_gradient,
    learning_rate=0.05,
    iterations=100
)

plt.figure(figsize=(10, 6))

plt.plot(gd_loss, label="Gradient Descent")
plt.plot(losses_m, label="Momentum")
plt.plot(losses_a, label="AdaGrad")
plt.plot(losses_r, label="RMSProp")
plt.plot(losses_adam, label="Adam")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Loss (log scale)")
plt.title("Optimizer Comparison")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 14. OPTIMIZER TRAJECTORY COMPARISON
# ============================================================

plt.figure(figsize=(10, 8))

contour = plt.contour(X, Y, Z, levels=30)

plt.plot(
    trajectory_gd[:, 0],
    trajectory_gd[:, 1],
    label="GD"
)

plt.plot(
    trajectory_m[:, 0],
    trajectory_m[:, 1],
    label="Momentum"
)

plt.plot(
    trajectory_a[:, 0],
    trajectory_a[:, 1],
    label="AdaGrad"
)

plt.plot(
    trajectory_r[:, 0],
    trajectory_r[:, 1],
    label="RMSProp"
)

plt.plot(
    trajectory_adam[:, 0],
    trajectory_adam[:, 1],
    label="Adam"
)

plt.scatter(0, 0, s=100, label="Minimum")

plt.xlabel("Parameter θ₁")
plt.ylabel("Parameter θ₂")
plt.title("Optimization Trajectories")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 15. REGULARIZED OBJECTIVE
# ============================================================

def ridge_objective(theta, X, y, lam):
    predictions = X @ theta
    mse = np.mean((predictions - y) ** 2)
    penalty = lam * np.sum(theta ** 2)
    return mse + penalty


def ridge_gradient(theta, X, y, lam):
    predictions = X @ theta
    error = predictions - y

    mse_gradient = (2 / len(X)) * X.T @ error
    penalty_gradient = 2 * lam * theta

    return mse_gradient + penalty_gradient


print("\n" + "=" * 70)
print("REGULARIZED OBJECTIVE")
print("=" * 70)

theta = np.zeros(X_train.shape[1])

for lam in [0.0, 0.001, 0.01, 0.1]:
    for _ in range(500):
        gradient = ridge_gradient(
            theta,
            X_train,
            y_train,
            lam
        )
        theta -= 0.01 * gradient

    print(
        f"lambda={lam:0.3f}, "
        f"objective={ridge_objective(theta, X_train, y_train, lam):.6f}"
    )


# ============================================================
# 16. TINY NEURAL NETWORK BRIDGE
# ============================================================

print("\n" + "=" * 70)
print("NEURAL NETWORK TRAINING BRIDGE")
print("=" * 70)

print("""
A neural network follows the same fundamental learning loop:

Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Parameter Update
  ↓
Repeat
""")

print("""
The important conceptual connection:

Regression:
    parameters → prediction → loss → gradient → update

Neural network:
    weights → forward pass → loss → backpropagation
    → gradients → optimizer → update
""")


# ============================================================
# 17. DAY 2 TAKEAWAY
# ============================================================

print("=" * 70)
print("DAY 2 TAKEAWAY")
print("=" * 70)

print("""
Today we opened the optimizer.

1. A loss function defines what we want to minimize.
2. A gradient tells us how the loss changes.
3. Gradient Descent moves against the gradient.
4. Learning rate controls update size.
5. Poor conditioning can cause slow or zig-zagging optimization.
6. Momentum uses previous movement to improve the trajectory.
7. AdaGrad adapts learning rates using accumulated gradients.
8. RMSProp uses an exponentially weighted gradient scale.
9. Adam combines momentum-like first-moment tracking with
   adaptive second-moment scaling.
10. Regularization changes the objective itself.
11. Neural networks use the same basic optimization loop,
    but with many more parameters and a more complex computation graph.
""")

print("Next: Day 3 - Perceptron, Classification and Neural Networks")
