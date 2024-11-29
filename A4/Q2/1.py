import pandas as pd

train_data = pd.read_csv('../dataset/train.csv')

print("Dataset Overview:")
print(train_data.info())

print("\nUnique Values in Each Column:")
unique_values = train_data.nunique()
print(unique_values)

print("\nSample Data:")
print(train_data.head())

numerical_columns = train_data.select_dtypes(include=['float64', 'int64'])

print("\nStatistical Analysis of Numerical Columns:")
stats = numerical_columns.describe(percentiles=[0.25, 0.5, 0.75]).T
stats['std'] = numerical_columns.std()
print(stats)

stats.to_csv('numerical_column_stats.csv', index=True)