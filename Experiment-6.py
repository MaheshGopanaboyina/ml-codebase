import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample dataset
data = {'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red', 'Green']}
df = pd.DataFrame(data)

# Apply Label Encoding
label_encoder = LabelEncoder()
df['Color_Encoded'] = label_encoder.fit_transform(df['Color'])

# Apply One-Hot Encoding
df_one_hot = pd.get_dummies(df, columns=['Color'], prefix='Color')

print("Original Data:")
print(df)
print("\nOne-Hot Encoded Data:")
print(df_one_hot)
