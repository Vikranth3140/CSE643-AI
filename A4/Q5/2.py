import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
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
dt = DecisionTreeRegressor(
    random_state=42,
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    ccp_alpha=31941543.737055868
)
dt.fit(X_train, y_train)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

dt_r2 = r2_score(y_test, dt_pred)
rf_r2 = r2_score(y_test, rf_pred)

dt_mse = mean_squared_error(y_test, dt_pred)
rf_mse = mean_squared_error(y_test, rf_pred)

dt_mae = mean_absolute_error(y_test, dt_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)

print(f"\nDecision Tree R2 Score: {dt_r2}")
print(f"Random Forest R2 Score: {rf_r2}")

print(f"\nDecision Tree Mean Squared Error: {dt_mse}")
print(f"Random Forest Mean Squared Error: {rf_mse}")

print(f"\nDecision Tree Mean Absolute Error: {dt_mae}")
print(f"Random Forest Mean Absolute Error: {rf_mae}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, dt_pred, c='red', label='Decision Tree')
plt.scatter(y_test, rf_pred, c='blue', label='Random Forest')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices')
plt.legend()

output_path = "Plots/predicted_vs_actual_prices.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()