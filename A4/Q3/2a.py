import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../Q2/X_train_final_with_categories.csv')
# train_data = pd.read_csv('../Q2/undersampled_train_data.csv')
# train_data = pd.read_csv('../Q2/oversampled_train_data.csv')

X_train = train_data.drop(columns=['Price', 'Price_Category'])
y_train = train_data['Price']

# Training Decision Tree on processed data
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Extract feature importances from the trained Decision Tree model
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)

plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Feature Importances from Decision Tree")
plt.xlabel("Features")
plt.ylabel("Importance")

output_path = "Plots/feature_importances.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()