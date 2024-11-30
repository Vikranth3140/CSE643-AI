import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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

# Identify Top 3 Important Features
feature_importances = model.feature_importances_
important_features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)
top_3_features = important_features.head(3).index
print("Top 3 Important Features:\n", important_features.head(3))

for feature in top_3_features:
    plt.figure(figsize=(8, 6))
    plt.scatter(train_data[feature], train_data['Price'], alpha=0.6, edgecolor='k')
    plt.title(f"Effect of {feature} on Price")
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.grid()
    output_path = f"Plots/{feature}_vs_Price.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()

# Calculate RMSE for Each Feature
for feature in top_3_features:
    X_feature = train_data[[feature]]
    X_train_feat, X_test_feat, y_train_feat, y_test_feat = train_test_split(X_feature, y, test_size=0.2, random_state=42)

    feature_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2
    )
    feature_model.fit(X_train_feat, y_train_feat)

    # Predict and calculate RMSE
    y_pred_feat = feature_model.predict(X_test_feat)
    rmse = np.sqrt(mean_squared_error(y_test_feat, y_pred_feat))
    print(f"RMSE when using only {feature}: {rmse}")