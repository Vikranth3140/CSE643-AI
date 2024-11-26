import pandas as pd

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Overview of the dataset
print("Dataset Overview:")
print(train_data.info())  # Provides data types and non-null counts

# Display unique values in each column
print("\nUnique Values in Each Column:")
unique_values = train_data.nunique()
print(unique_values)

# Display the first few rows of the dataset for reference
print("\nSample Data:")
print(train_data.head())

# Select numerical columns
numerical_columns = train_data.select_dtypes(include=['float64', 'int64'])

# Perform statistical analysis
print("\nStatistical Analysis of Numerical Columns:")
stats = numerical_columns.describe(percentiles=[0.25, 0.5, 0.75]).T  # Transpose for readability
stats['std'] = numerical_columns.std()  # Add standard deviation explicitly
print(stats)

# Optionally save the statistics to a file for reference
stats.to_csv('numerical_column_stats.csv', index=True)