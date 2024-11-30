import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error

# train_data = pd.read_csv('../dataset/train.csv')
train_data = pd.read_csv('../processed_train_data.csv')

X = train_data.drop(columns=['Price'])
y = train_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Decision Tree on processed data using the Best Hyperparameters found in 2b
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=10,
    max_features=None,
    min_samples_leaf=2,
    min_samples_split=2
)
model.fit(X_train, y_train)



# Get effective alphas and corresponding total leaf impurities for pruning
path = model.cost_complexity_pruning_path(X_train, y_train)
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

# Print the optimal ccp_alpha and its corresponding test error
print(f"Optimal ccp_alpha: {best_alpha}")
print(f"Mean Squared Error (Pruned Tree): {test_scores[best_alpha_index]}")


# Visualize the original (unpruned) tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Unpruned Decision Tree")
plt.show()


# Visualize the pruned tree
plt.figure(figsize=(20, 10))
plot_tree(pruned_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Pruned Decision Tree")
plt.show()