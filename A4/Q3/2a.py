import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Extract feature importances from the trained Decision Tree model
feature_importances = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Feature Importances from Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")

output_path = "Plots/feature_importances.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()