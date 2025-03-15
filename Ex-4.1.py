from sklearn.linear_model import LinearRegression
import numpy as np

# Example dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Model initialization and training
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)
print(predictions)
