from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree using the ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy')  # ID3 uses entropy
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the decision tree
print("Decision Tree:")
print(export_text(clf, feature_names=iris.feature_names))

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classify a new sample
def classify_new_sample(sample):
    prediction = clf.predict([sample])
    return iris.target_names[prediction[0]]

# Example of a new sample
new_sample = [5.1, 3.5, 1.4, 0.2]  # Example flower features
print(f"\nNew Sample Classification: {classify_new_sample(new_sample)}")
