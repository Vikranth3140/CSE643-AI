import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
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

# Perform 5-fold cross-validation
# We use Negative MSE as the scoring metric
cv_mse = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='neg_mean_squared_error'
)
cv_mse = -cv_mse
print("Cross-Validation MSE Scores:", cv_mse)
print("Mean CV MSE:", np.mean(cv_mse))

# Implement Learning Curves
train_sizes, train_scores, validation_scores = learning_curve(
    model,
    X,
    y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42
)

train_errors = -train_scores.mean(axis=1)
validation_errors = -validation_scores.mean(axis=1)

# Plot Learning Curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_errors, label="Training Error (MSE)", marker="o")
plt.plot(train_sizes, validation_errors, label="Validation Error (MSE)", marker="o")
plt.title("Learning Curves: Decision Tree (MSE)")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()

output_path = "Plots/learning_curves_decision_tree_mse.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Test R^2 Score: {r2}")
print(f"Test Mean Squared Error: {mse}")