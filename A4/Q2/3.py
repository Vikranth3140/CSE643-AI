import pandas as pd
from sklearn.preprocessing import LabelEncoder

# train_data = pd.read_csv('../dataset/train.csv')
train_data = pd.read_csv('dropped_cols_train_data.csv')

# Here, we assume that all the columns with 'object' data type are categorical columns, including "Address"
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

print("\nTransformed Dataset (with Encoded Categorical Columns):")
print(train_data.head())