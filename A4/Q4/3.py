import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
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
# `ccp_alpha` = 31941543.737055868 is the best hyperparameter
model = DecisionTreeRegressor(
    random_state=42,
    max_depth=None,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    ccp_alpha=31941543.737055868
)
model.fit(X_train, y_train)

# Identify Top 3 Important Features
feature_importances = model.feature_importances_
important_features = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)
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
    X_train_feature = X_train[[feature]]
    X_test_feature = X_test[[feature]]

    feature_model = DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_leaf=2,
        min_samples_split=2
    )
    feature_model.fit(X_train_feature, y_train)

    y_pred_feature = feature_model.predict(X_test_feature)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_feature))
    print(f"RMSE when using only {feature}: {rmse}")