import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from numpy.linalg import inv

# Generate dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X = X.flatten()

def kernel(x, x_i, tau):
    return np.exp(-((x - x_i) ** 2) / (2 * tau ** 2))

def locally_weighted_regression(x_test, X_train, y_train, tau=0.5):
    m = X_train.shape[0]
    y_pred = np.zeros_like(x_test)
    
    for i, x in enumerate(x_test):
        W = np.diag(kernel(x, X_train, tau))  # Compute weights
        theta = inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train
        y_pred[i] = np.array([1, x]) @ theta
    
    return y_pred

# Add intercept term
X_train = np.column_stack((np.ones_like(X), X))
X_test = np.linspace(min(X), max(X), 100)
X_test_bias = np.column_stack((np.ones_like(X_test), X_test))

# Fit model
y_pred = locally_weighted_regression(X_test, X_train, y, tau=0.1)

# Plot results
plt.scatter(X, y, label="Data Points", color="blue", alpha=0.5)
plt.plot(X_test, y_pred, label="LWR Fit", color="red")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.title("Locally Weighted Regression")
plt.show()
