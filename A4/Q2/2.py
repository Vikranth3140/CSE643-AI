import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('../dataset/train.csv')

# Select only numerical columns for correlation analysis
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numerical_data.corr()

# Display the correlation of each feature with the target variable
# Replace 'target_variable' with your actual target column name
if 'target_variable' in correlation_matrix.columns:
    target_correlation = correlation_matrix['target_variable']
    print("\nCorrelation with Target Variable:")
    print(target_correlation)

    # Identify columns with weak correlation in the range [-0.1, 0.1]
    weak_correlation_columns = target_correlation[(target_correlation > -0.1) & (target_correlation < 0.1)].index
    print("\nColumns to Drop (Weak Correlation):")
    print(weak_correlation_columns)

    # Drop these columns
    data = data.drop(columns=weak_correlation_columns)

    # Justify removal of each column
    for column in weak_correlation_columns:
        print(f"Column '{column}' was dropped due to weak correlation ({target_correlation[column]:.2f}).")
else:
    print("Target variable not found in the dataset.")

# Display the correlation of each feature with the target variable
target_correlation = correlation_matrix['target_variable']  # Replace 'target_variable' with the actual column name
print("\nCorrelation with Target Variable:")
print(target_correlation)

# Visualize the correlation matrix (optional)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Identify columns with correlation in the range [-0.1, 0.1]
weak_correlation_columns = target_correlation[(target_correlation > -0.1) & (target_correlation < 0.1)].index
print("\nColumns to Drop (Weak Correlation):")
print(weak_correlation_columns)

# Drop these columns
data = data.drop(columns=weak_correlation_columns)

# Justify removal of each column
for column in weak_correlation_columns:
    print(f"Column '{column}' was dropped due to weak correlation ({target_correlation[column]:.2f}).")

# Drop columns that are non-predictive (e.g., IDs, placeholders)
non_predictive_columns = ['id_column']  # Replace 'id_column' with actual column names
data = data.drop(columns=non_predictive_columns)

for column in non_predictive_columns:
    print(f"Column '{column}' was dropped as it does not contribute meaningfully to prediction.")