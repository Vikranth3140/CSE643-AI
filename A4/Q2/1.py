import pandas as pd

train_data = pd.read_csv('../dataset/train.csv')
test_data = pd.read_csv('../dataset/test.csv')

# Train Dataset
print("Train Dataset Overview:")
print(train_data.info())

print("\nTrain Unique Values in Each Column:")
unique_values = train_data.nunique()
print(unique_values)

print("\nTrain Sample Data:")
print(train_data.head())

numerical_columns_train = train_data.select_dtypes(include=['float64', 'int64'])

print("\nTrain Statistical Analysis of Numerical Columns:")
stats = numerical_columns_train.describe(percentiles=[0.25, 0.5, 0.75]).T
stats['std'] = numerical_columns_train.std()
print(stats)

stats.to_csv('numerical_column_stats_train.csv', index=True)


# Test Dataset
print("Test Dataset Overview:")
print(test_data.info())

print("\nTest Unique Values in Each Column:")
unique_values = test_data.nunique()
print(unique_values)

print("\nTest Sample Data:")
print(test_data.head())

numerical_columns_test = test_data.select_dtypes(include=['float64', 'int64'])

print("\nTest Statistical Analysis of Numerical Columns:")
stats = numerical_columns_test.describe(percentiles=[0.25, 0.5, 0.75]).T
stats['std'] = numerical_columns_test.std()
print(stats)

stats.to_csv('numerical_column_stats_test.csv', index=True)