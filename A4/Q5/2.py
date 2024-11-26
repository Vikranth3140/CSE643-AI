from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
train_data = pd.read_csv('../dataset/train.csv')

# Create 'Price_Category' if not already created for other tasks
def categorize_price(price):
    if price < train_data['Price'].quantile(0.25):  # Bottom 25%
        return 'Low'
    elif price < train_data['Price'].quantile(0.5):  # 25% to 50%
        return 'Medium'
    elif price < train_data['Price'].quantile(0.75):  # 50% to 75%
        return 'High'
    else:  # Top 25%
        return 'Very High'

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

# Prepare features (X) and target (y)
X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])
y = train_data['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions with Decision Tree
y_train_pred_dt = dt_regressor.predict(X_train)
y_test_pred_dt = dt_regressor.predict(X_test)

# Make predictions with Random Forest
y_train_pred_rf = rf_regressor.predict(X_train)
y_test_pred_rf = rf_regressor.predict(X_test)

# Evaluate Decision Tree
dt_train_mse = mean_squared_error(y_train, y_train_pred_dt)
dt_train_mae = mean_absolute_error(y_train, y_train_pred_dt)
dt_train_r2 = r2_score(y_train, y_train_pred_dt)

dt_test_mse = mean_squared_error(y_test, y_test_pred_dt)
dt_test_mae = mean_absolute_error(y_test, y_test_pred_dt)
dt_test_r2 = r2_score(y_test, y_test_pred_dt)

print("Decision Tree Regressor Performance:")
print(f"Training MSE: {dt_train_mse:.2f}, Training MAE: {dt_train_mae:.2f}, Training R²: {dt_train_r2:.2f}")
print(f"Test MSE: {dt_test_mse:.2f}, Test MAE: {dt_test_mae:.2f}, Test R²: {dt_test_r2:.2f}\n")

# Evaluate Random Forest
rf_train_mse = mean_squared_error(y_train, y_train_pred_rf)
rf_train_mae = mean_absolute_error(y_train, y_train_pred_rf)
rf_train_r2 = r2_score(y_train, y_train_pred_rf)

rf_test_mse = mean_squared_error(y_test, y_test_pred_rf)
rf_test_mae = mean_absolute_error(y_test, y_test_pred_rf)
rf_test_r2 = r2_score(y_test, y_test_pred_rf)

print("Random Forest Regressor Performance:")
print(f"Training MSE: {rf_train_mse:.2f}, Training MAE: {rf_train_mae:.2f}, Training R²: {rf_train_r2:.2f}")
print(f"Test MSE: {rf_test_mse:.2f}, Test MAE: {rf_test_mae:.2f}, Test R²: {rf_test_r2:.2f}\n")