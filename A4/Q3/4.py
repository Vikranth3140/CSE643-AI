import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import shutil
import tempfile
temp_folder = tempfile.gettempdir()
shutil.rmtree(temp_folder, ignore_errors=True)


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

import matplotlib.pyplot as plt
import pandas as pd

# Extract feature importances from the trained Decision Tree model
feature_importances = pd.Series(dt_regressor.feature_importances_, index=X.columns)

# Plot feature importances
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Feature Importances from Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

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



from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error

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



import matplotlib.pyplot as plt

# Plot training and testing errors vs ccp_alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, marker='o', label='Training Error')
plt.plot(ccp_alphas, test_scores, marker='o', label='Testing Error')
plt.xlabel('ccp_alpha')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Minimal Cost-Complexity Pruning')
plt.legend()
plt.grid()
plt.show()


# Select the best tree (minimum testing error)
best_alpha_index = test_scores.index(min(test_scores))
best_alpha = ccp_alphas[best_alpha_index]
pruned_tree = trees[best_alpha_index]



from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(dt_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Calculate mean and standard deviation of cross-validation scores
mean_cv_score = -cv_scores.mean()
std_cv_score = cv_scores.std()

print(f"Mean CV MSE: {mean_cv_score:.2f}")
print(f"Standard Deviation of CV MSE: {std_cv_score:.2f}")


from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    dt_regressor, X_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# Calculate mean and standard deviation for training and validation scores
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = -np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Training Error", marker='o')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, label="Validation Error", marker='o')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

plt.title("Learning Curves")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()
plt.show()