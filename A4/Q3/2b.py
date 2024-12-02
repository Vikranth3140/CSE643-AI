import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

train_data = pd.read_csv('../Q2/X_train_final_with_categories.csv')
test_data = pd.read_csv('../Q2/X_test_final_with_categories.csv')

X_train = train_data.drop(columns=['Price', 'Price_Category'])
y_train = train_data['Price']

X_test = test_data.drop(columns=['Price', 'Price_Category'])
y_test = test_data['Price']

# Hyperparameter Optimization using Grid Search

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2']
}

# Training Decision Tree on processed data
model = DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters from Grid Search:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score: {r2}")
print(f"Mean Squared Error (Tuned Model): {mse}")