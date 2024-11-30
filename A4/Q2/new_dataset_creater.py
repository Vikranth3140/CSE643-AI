import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../dataset/train.csv')

numerical_data = train_data.select_dtypes(include=['float64', 'int64'])

correlation_matrix = numerical_data.corr()

if 'Price' in correlation_matrix.columns:
    target_correlation = correlation_matrix['Price']
    print("\nCorrelation with Target Variable:")
    print(target_correlation)

    weak_correlation_columns = target_correlation[(target_correlation > -0.1) & (target_correlation < 0.1)].index
    print("\nColumns to Drop Due to Weak Correlation:")
    print(weak_correlation_columns)

    train_data = train_data.drop(columns=weak_correlation_columns)

def categorize_price(price):
    if price < train_data['Price'].quantile(0.25):
        return 'Low'
    elif price < train_data['Price'].quantile(0.5):
        return 'Medium'
    elif price < train_data['Price'].quantile(0.75):
        return 'High'
    else:
        return 'Very High'

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

categorical_columns = train_data.select_dtypes(include=['object']).columns

# Label encoding the categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])

numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Scaling the numerical columns
scaler = StandardScaler()

scaled_features = scaler.fit_transform(train_data[numerical_columns])

train_data[numerical_columns] = scaled_features

output_csv_path = "../processed_train_data.csv"
train_data.to_csv(output_csv_path, index=False)

print("\nModified dataset with 'Price_Category', dropped irrelevant columns, label-encoded features, standard scaled numerical values saved to:", output_csv_path)