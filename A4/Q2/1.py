import pandas as pd

# Load the dataset
data = pd.read_csv('../dataset/train.csv')

# Display dataset overview
print("Dataset Overview:")
print(data.info())  # Overview of columns, data types, and non-null counts

# Summarize unique values in each column
print("\nUnique Values in Each Column:")
print(data.nunique())

# Display first few rows for a quick look
print("\nSample Data:")
print(data.head())


# Select numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64'])

# Perform detailed statistical analysis
print("\nStatistical Analysis of Numerical Columns:")
stats = numerical_cols.describe(percentiles=[0.25, 0.5, 0.75]).T  # Transpose for better readability
stats['std'] = numerical_cols.std()  # Add standard deviation explicitly
print(stats)

# Save stats to a CSV file (optional)
stats.to_csv('numerical_stats.csv', index=True)