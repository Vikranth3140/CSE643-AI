import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
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
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2
)
model.fit(X_train, y_train)

# Get effective alphas and corresponding total leaf impurities for pruning
path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

# Train a series of Decision Trees with different ccp_alpha values
models = []
train_mse = []
test_mse = []

for alpha in ccp_alphas:
    pruned_model = DecisionTreeRegressor(random_state=42, ccp_alpha=alpha)
    pruned_model.fit(X_train, y_train)
    models.append(pruned_model)
    train_mse.append(mean_squared_error(y_train, pruned_model.predict(X_train)))
    test_mse.append(mean_squared_error(y_test, pruned_model.predict(X_test)))

# Plot training and testing errors vs ccp_alpha
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_mse, label='Train MSE', marker='o')
plt.plot(ccp_alphas, test_mse, label='Test MSE', marker='o')
plt.xlabel('ccp_alpha')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Minimal Cost-Complexity Pruning on MSE')
plt.legend()
plt.grid()

output_path = "Plots/pruning_effect_mse_scores.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

# Select the best tree (minimum testing error)
optimal_alpha = ccp_alphas[test_mse.index(min(test_mse))]
print(f"Optimal ccp_alpha: {optimal_alpha}")

# Retrain the Tree with Optimal ccp_alpha
pruned_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    ccp_alpha=optimal_alpha
)
pruned_model.fit(X_train, y_train)

y_pred = pruned_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
final_mse = mean_squared_error(y_test, y_pred)

print(f"Pruned Model R2 Score: {r2}")
print(f"Pruned Model Mean Squared Error: {final_mse}")

plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X_train.columns, rounded=True)
plt.title("Decision Tree Before Pruning")

output_path = "Plots/unpruned_decision_tree.png"
plt.savefig(output_path, bbox_inches='tight')
# plt.show()

plt.figure(figsize=(20, 10))
plot_tree(pruned_model, filled=True, feature_names=X_train.columns, rounded=True)
plt.title("Decision Tree After Pruning")

output_path = "Plots/pruned_decision_tree.png"
plt.savefig(output_path, bbox_inches='tight')
# plt.show()