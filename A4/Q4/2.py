import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../Q2/X_train_final_with_categories.csv')
test_data = pd.read_csv('../Q2/X_test_final_with_categories.csv')

X_train = train_data.drop(columns=['Price', 'Price_Category'])
y_train = train_data['Price']

X_test = test_data.drop(columns=['Price', 'Price_Category'])
y_test = test_data['Price']

# Training Decision Tree on processed data using the Best Hyperparameters found in 2b
# `ccp_alpha` = 31941543.737055868 is the best hyperparameter
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    ccp_alpha=31941543.737055868
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print("Training Residuals Summary:")
print(train_residuals.describe())

print("\nTest Residuals Summary:")
print(test_residuals.describe())

plt.figure(figsize=(10, 6))
plt.hist(test_residuals, bins=30, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
plt.title("Distribution of Test Residuals")
plt.xlabel("Residuals (Actual - Predicted)")
plt.ylabel("Frequency")
plt.legend()

output_path = "Plots/distribution_of_test_residuals.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, test_residuals, alpha=0.6, edgecolor='black')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.title("Residuals vs. Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.grid(True)

output_path = "Plots/residuals_vs_predicted_prices.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(X_test['Carpet_area'], test_residuals, alpha=0.6, edgecolor='black')
plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
plt.title("Residuals vs. Carpet Area")
plt.xlabel("Carpet Area")
plt.ylabel("Residuals (Actual - Predicted)")
plt.legend()
plt.grid(True)

output_path = "Plots/residuals_vs_carpet_size.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()