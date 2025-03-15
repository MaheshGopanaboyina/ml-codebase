import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# Generate Synthetic Data
def generate_data(n=100, noise=0.1):
    np.random.seed(42)
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = np.sin(X) + noise * np.random.randn(n, 1)
    return X, y

X, y = generate_data()

# Remove Duplicates from DataFrame
def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

data = {'A': [1, 2, 2, 3], 'B': [10, 20, 20, 30]}
df = pd.DataFrame(data)
df = remove_duplicates(df)
print("Cleaned DataFrame:\n", df)

# Perform K-Fold Cross-Validation
def cross_validation(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y.ravel(), cv=kf, scoring='neg_mean_squared_error')
    print(f"Mean MSE: {-scores.mean():.4f}, Std Dev: {scores.std():.4f}")

model = LinearRegression()
cross_validation(model, X, y)
