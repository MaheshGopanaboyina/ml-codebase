import pandas as pd
import numpy as np

# Define the file path for the dataset
file_path = r"C:\Users\CSELAB2\Downloads\data (1).csv"

# Load the dataset into a Pandas DataFrame
data = pd.read_csv(file_path)
print("Training Data:")
print(data)

# Extract the attribute values (features) excluding the target column
concepts = np.array(data)[:, :-1]
print("Concepts:")
print(concepts)

# Extract the target values (last column)
target = np.array(data)[:, -1]
print("Target:")
print(target)

def train(con, tar):
    """
    Function to implement the FIND-S algorithm.
    It finds the most specific hypothesis that fits positive examples.
    """
    for i, val in enumerate(tar):
        if val.lower() == 'yes':  # Ensure case insensitivity for positive examples
            specific_h = con[i].copy()  # Initialize with the first positive example
            break
    return specific_h 

# Train the model and print the most specific hypothesis
print("Most Specific Hypothesis:", train(concepts, target))
