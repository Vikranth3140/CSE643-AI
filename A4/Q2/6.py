import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

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
columns_to_exclude = ['Price', 'Address', 'Furnishing']
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

X_train_final['Price_Category'] = X_train_final['Price'].apply(categorize_price)
X_test_final['Price_Category'] = X_test_final['Price'].apply(categorize_price)

# Train Data
X_train = X_train_final.drop(columns=['Price_Category'])
y_train = X_train_final['Price_Category']

print("Original Training Set Distribution (Train Data):")
print(y_train.value_counts())

# Apply undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

undersampled_data = pd.concat([X_train_under, y_train_under], axis=1)
undersampled_data.to_csv('undersampled_train_data.csv', index=False)

print("\nDistribution After Random Undersampling (Train Data):")
print(y_train_under.value_counts())

# Apply oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

oversampled_data = pd.concat([X_train_over, y_train_over], axis=1)
oversampled_data.to_csv('oversampled_train_data.csv', index=False)

print("\nDistribution After Random Oversampling (Train Data):")
print(y_train_over.value_counts())

plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_under.value_counts().index, y=y_train_under.value_counts().values, palette='viridis')
plt.title('Distribution After Random Undersampling (Train Data)')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.savefig("Plots/undersampling_train_data_distribution.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_over.value_counts().index, y=y_train_over.value_counts().values, palette='viridis')
plt.title('Distribution After Random Oversampling (Train Data)')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.savefig("Plots/oversampling_train_data_distribution.png", bbox_inches='tight')
plt.show()


# Test Data
X_test = X_test_final.drop(columns=['Price_Category'])
y_test = X_test_final['Price_Category']

print("Original Training Set Distribution (Test Data):")
print(y_test.value_counts())

# Apply undersampling
X_test_under, y_test_under = undersampler.fit_resample(X_test, y_test)

undersampled_data = pd.concat([X_test_under, y_test_under], axis=1)
undersampled_data.to_csv('undersampled_test_data.csv', index=False)

print("\nDistribution After Random Undersampling (Test Data):")
print(y_test_under.value_counts())

# Apply oversampling
X_test_over, y_test_over = oversampler.fit_resample(X_test, y_test)

oversampled_data = pd.concat([X_test_over, y_test_over], axis=1)
oversampled_data.to_csv('oversampled_test_data.csv', index=False)

print("\nDistribution After Random Oversampling (Test Data):")
print(y_test_over.value_counts())

plt.figure(figsize=(8, 6))
sns.barplot(x=y_test_under.value_counts().index, y=y_test_under.value_counts().values, palette='viridis')
plt.title('Distribution After Random Undersampling (Test Data)')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.savefig("Plots/undersampling_test_data_distribution.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=y_test_over.value_counts().index, y=y_test_over.value_counts().values, palette='viridis')
plt.title('Distribution After Random Oversampling (Test Data)')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.savefig("Plots/oversampling_test_data_distribution.png", bbox_inches='tight')
plt.show()