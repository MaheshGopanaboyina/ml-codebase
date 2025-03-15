import pandas as pd
import math

# Function to calculate entropy
def calculate_entropy(data):
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts()
    total_samples = len(labels)
    entropy = sum((-count / total_samples) * math.log2(count / total_samples) for count in label_counts)
    return entropy

# Split data based on an attribute and value
def split_data(data, attribute, value):
    return data[data[attribute] == value].reset_index(drop=True)

# Function to calculate information gain
def information_gain(data, attribute):
    total_entropy = calculate_entropy(data)
    values = data[attribute].unique()
    subset_entropy = 0
    
    for value in values:
        subset = split_data(data, attribute, value)
        subset_entropy += (len(subset) / len(data)) * calculate_entropy(subset)
    
    return total_entropy - subset_entropy

# Function to build the decision tree
def build_tree(data, features):
    labels = data.iloc[:, -1]
    if len(labels.unique()) == 1:
        return labels.iloc[0]
    
    features = list(features)  # Convert to list to avoid ambiguity
    if not features:
        return labels.mode()[0]
    
    # Find the best feature to split on
    gains = {feature: information_gain(data, feature) for feature in features}
    best_feature = max(gains, key=gains.get)
    
    # Create the tree
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]
    
    for value in data[best_feature].unique():
        subtree_data = split_data(data, best_feature, value)
        tree[best_feature][value] = build_tree(subtree_data, remaining_features)
    
    return tree

# Function to classify a new sample using the decision tree
def classify(sample, tree):
    if not isinstance(tree, dict):
        return tree
    
    feature = next(iter(tree))
    value = sample.get(feature, None)
    
    if value not in tree[feature]:
        return None  # Classification cannot be determined
    return classify(sample, tree[feature][value])

# Example dataset
file=r"C:\Users\CSELAB2\Downloads\PlayTennis.csv"
df=pd.read_csv(file)


# Train the decision tree
features = df.columns[:-1]  # Exclude the target column
tree = build_tree(df, features)
print("Trained Decision Tree:")
print(tree)

# Classify a new sample
new_sample = {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "High", "Windy": "True"}
result = classify(new_sample, tree)
print("\nNew Sample Classification:")
print(f"Sample: {new_sample}, Classified as: {result}")
