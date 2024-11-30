import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
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