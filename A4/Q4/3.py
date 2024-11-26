import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
train_data = pd.read_csv('../dataset/train.csv')

# Define price brackets for target variable
def categorize_price(price):
    if price < 10000000:
        return 'Low'
    elif 10000000 <= price < 25000000:
        return 'Medium'
    elif 25000000 <= price < 50000000:
        return 'High'
    else:
        return 'Very High'

# Create Price_Category column
train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

# Prepare features (X) and target (y)
X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])  # Drop categorical and target columns
y = train_data['Price']  # Use Price for regression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Extract feature importances from the trained Decision Tree model
feature_importances = pd.Series(dt_regressor.feature_importances_, index=X.columns)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Feature Importances from Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()



# Define the parameter grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2']
}

# Initialize a Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)

# Initialize Grid Search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
print("Best Parameters from Grid Search:")
print(grid_search.best_params_)

# Train model with the best parameters
best_dt = grid_search.best_estimator_

# Evaluate the tuned model on the test set
y_pred_tuned = best_dt.predict(X_test)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)

print(f"\nMean Squared Error (Tuned Model): {mse_tuned}")



# Get effective alphas and corresponding total leaf impurities for pruning
path = dt_regressor.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Train a series of Decision Trees with different ccp_alpha values
trees = []
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeRegressor(random_state=42, ccp_alpha=ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Evaluate pruned trees using their training and testing errors
train_scores = [mean_squared_error(y_train, tree.predict(X_train)) for tree in trees]
test_scores = [mean_squared_error(y_test, tree.predict(X_test)) for tree in trees]


# Extract feature importances

feature_importances = pd.Series(best_dt.feature_importances_, index=X.columns).sort_values(ascending=False)
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