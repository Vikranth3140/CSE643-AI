import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Select only numerical columns for correlation analysis
numerical_data = train_data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix for numerical columns only
correlation_matrix = numerical_data.corr()

# Correlation with the target variable (assumed 'Price')
if 'Price' in correlation_matrix.columns:
    target_correlation = correlation_matrix['Price']
    print("\nCorrelation with Target Variable:")
    print(target_correlation)

    # Identify columns with weak correlation in the range [-0.1, 0.1]
    weak_correlation_columns = target_correlation[(target_correlation > -0.1) & (target_correlation < 0.1)].index
    print("\nColumns to Drop Due to Weak Correlation:")
    print(weak_correlation_columns)

    # Drop these columns from the dataset
    train_data = train_data.drop(columns=weak_correlation_columns)

    # Provide justification for each dropped column
    for column in weak_correlation_columns:
        print(f"Column '{column}' was dropped due to weak correlation ({target_correlation[column]:.2f}) with 'Price'.")
else:
    print("Target variable 'Price' is not found in the correlation matrix.")

# Optional: Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()