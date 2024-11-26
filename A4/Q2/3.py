import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Identify categorical columns
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

# Display sample data after encoding
print("\nTransformed Dataset (with Encoded Categorical Columns):")
print(train_data.head())