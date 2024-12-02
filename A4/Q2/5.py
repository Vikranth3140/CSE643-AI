import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("Plots", exist_ok=True)

# train_data = pd.read_csv('../dataset/train.csv')
# test_data = pd.read_csv('../dataset/test.csv')
train_data = pd.read_csv('dropped_cols_train_data.csv')
test_data = pd.read_csv('dropped_cols_test_data.csv')

# First we encode the categorical columns and then scale the numerical columns

# Here, we assume that all the columns with 'object' data type are categorical columns, including "Address"
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Label encoding the categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.fit_transform(test_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Scaled data
columns_to_exclude = ['Price', 'Address', 'Possesion', 'Furnishing']
numerical_columns_to_scale = [col for col in train_data.select_dtypes(include=['float64', 'int64']).columns if col not in columns_to_exclude]

# Apply StandardScaler to scale the features
scaler = StandardScaler()

X_train_scaled_numerical = scaler.fit_transform(train_data[numerical_columns_to_scale])
X_test_scaled_numerical = scaler.transform(test_data[numerical_columns_to_scale])

X_train_scaled_numerical_df = pd.DataFrame(X_train_scaled_numerical, columns=numerical_columns_to_scale)
X_test_scaled_numerical_df = pd.DataFrame(X_test_scaled_numerical, columns=numerical_columns_to_scale)

excluded_columns_train = train_data[columns_to_exclude].reset_index(drop=True)
excluded_columns_test = test_data[columns_to_exclude].reset_index(drop=True)

X_train_final = pd.concat([X_train_scaled_numerical_df, excluded_columns_train], axis=1)
X_test_final = pd.concat([X_test_scaled_numerical_df, excluded_columns_test], axis=1)


plt.figure(figsize=(10, 6))
plt.hist(train_data['Price'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y')

output_path = "Plots/price_distribution.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

def categorize_price(price):
    if price < 10000000:
        return 'Low'
    elif 10000000 <= price < 20000000:
        return 'Medium'
    elif 20000000 <= price < 40000000:
        return 'High'
    else:
        return 'Very High'

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)
test_data['Price_Category'] = test_data['Price'].apply(categorize_price)

print(train_data[['Price', 'Price_Category']].head())

category_counts = train_data['Price_Category'].value_counts()

# Plot the distribution of price categories
plt.figure(figsize=(8, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Distribution of Properties Across Price Categories')
plt.xlabel('Price Category')
plt.ylabel('Number of Properties')

output_path = "Plots/price_category_distribution.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

print("\nDistribution of Properties Across Price Categories:")
print(category_counts)