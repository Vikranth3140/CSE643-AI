import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../Q2/X_train_final_with_categories.csv')
test_data = pd.read_csv('../Q2/X_test_final_with_categories.csv')
# train_data = pd.read_csv('../Q2/undersampled_train_data.csv')
# test_data = pd.read_csv('../Q2/undersampled_test_data.csv')
# train_data = pd.read_csv('../Q2/oversampled_train_data.csv')
# test_data = pd.read_csv('../Q2/oversampled_test_data.csv')

X_train = train_data.drop(columns=['Price', 'Price_Category'])
y_train = train_data['Price']

X_test = test_data.drop(columns=['Price', 'Price_Category'])
y_test = test_data['Price']

# Training Decision Tree on processed data using the Best Hyperparameters found in 2b
# `ccp_alpha` = 31941543.737055868 is the best hyperparameter
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10,
    max_features=None,
    min_samples_leaf=2,
    min_samples_split=5,
    ccp_alpha=22217164.00379489
)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Train Performance:")
print(f"  R² Score: {r2_train:.4f}")
print(f"  Mean Squared Error: {mse_train:.4f}")
print(f"  Mean Absolute Error: {mae_train:.4f}")

print("\nTest Performance:")
print(f"  R² Score: {r2_test:.4f}")
print(f"  Mean Squared Error: {mse_test:.4f}")
print(f"  Mean Absolute Error: {mae_test:.4f}")