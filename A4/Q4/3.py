import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

# train_data = pd.read_csv('../dataset/train.csv')
train_data = pd.read_csv('../processed_train_data.csv')

X = train_data.drop(columns=['Price'])
y = train_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Decision Tree on processed data using the Best Hyperparameters found in 2b
# Do not need to prune the tree as `ccp_alpha` = 0 is the best hyperparameter
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10,
    max_features=None,
    min_samples_leaf=2,
    min_samples_split=2
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


# Extract feature importances

feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top_features = feature_importances.head(3).index
print("Top 3 Important Features:")
print(top_features)

# Visualize the relationship for each feature
for feature in top_features:
    plt.figure(figsize=(10, 6))
    plt.scatter(X[feature], y, alpha=0.6, edgecolor='black')
    plt.title(f"{feature} vs. Price")
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

# Initialize dictionary to store RMSE values
feature_rmse = {}

# Fit models and calculate RMSE
for feature in top_features:
    # Reshape feature for regression
    X_feature = X[[feature]]
    
    # Train-test split for individual feature
    X_train_feature, X_test_feature, y_train, y_test = train_test_split(X_feature, y, test_size=0.2, random_state=42)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train_feature, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_feature)
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    feature_rmse[feature] = rmse
    
    print(f"Feature: {feature}")
    print(f"RMSE: {rmse:.2f}")
    print()