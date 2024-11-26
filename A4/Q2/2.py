import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Compute correlation matrix for numerical columns
correlation_matrix = train_data.corr()

# Correlation with the target variable (assumed 'Price' here; replace if different)
target_correlation = correlation_matrix['Price']
print("\nCorrelation with Target Variable:")
print(target_correlation)

# Identify columns with weak correlation (-0.1 to 0.1)
weak_correlation_columns = target_correlation[(target_correlation > -0.1) & (target_correlation < 0.1)].index
print("\nColumns to Drop Due to Weak Correlation:")
print(weak_correlation_columns)

# Drop columns with weak correlation
train_data = train_data.drop(columns=weak_correlation_columns)

# Provide justification for dropping each column
for column in weak_correlation_columns:
    print(f"Column '{column}' was dropped due to weak correlation ({target_correlation[column]:.2f}) with 'Price'.")

# Drop non-predictive columns (e.g., 'index', 'Address')
non_predictive_columns = ['index', 'Address']  # Replace with actual columns you want to drop
train_data = train_data.drop(columns=non_predictive_columns)

# Provide justification for dropping each column
for column in non_predictive_columns:
    print(f"Column '{column}' was dropped as it does not contribute meaningfully to prediction.")

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()