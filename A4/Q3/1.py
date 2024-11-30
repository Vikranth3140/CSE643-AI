import pandas as pd
from sklearn.model_selection import train_test_split
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

# Training Decision Tree on processed data
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nR2 Score: {r2}")
print(f"\nMean Squared Error (Tuned Model): {mse}")

print(f"Decision Tree Depth: {model.get_depth()}")
print(f"Number of Leaves: {model.get_n_leaves()}")

plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure")

output_path = "Plots/decision_tree_structure.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()