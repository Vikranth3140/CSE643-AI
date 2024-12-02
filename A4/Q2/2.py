import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../dataset/train.csv')
test_data = pd.read_csv('../dataset/test.csv')

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
    test_data = test_data.drop(columns=weak_correlation_columns)

    for column in weak_correlation_columns:
        print(f"Column '{column}' was dropped due to weak correlation ({target_correlation[column]:.2f}) with 'Price'.")
else:
    print("Target variable 'Price' is not found in the correlation matrix.")

dropped_cols_data_path_train = "dropped_cols_train_data.csv"
dropped_cols_data_path_test = "dropped_cols_test_data.csv"
train_data.to_csv(dropped_cols_data_path_train, index=False)
test_data.to_csv(dropped_cols_data_path_test, index=False)
print(f"\nProcessed train data saved to: {dropped_cols_data_path_train}")
print(f"\nProcessed test data saved to: {dropped_cols_data_path_test}")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")

output_path = "Plots/correlation_matrix.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()