import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../dataset/train.csv')

# Here, we assume that all the columns with 'object' data type are categorical columns, including "Address"
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Scaling the numerical columns
scaler = StandardScaler()

scaled_features = scaler.fit_transform(train_data[numerical_columns])

train_data[numerical_columns] = scaled_features

print("\nScaled Numerical Features:")
print(train_data[numerical_columns].head())

# Training Decision Tree on scaled data
X_scaled = train_data.drop(columns=['Price'])
y_scaled = train_data['Price']
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

model_scaled = DecisionTreeRegressor(random_state=42)
model_scaled.fit(X_train_scaled, y_train_scaled)

print(f"Decision Tree Depth: {model_scaled.get_depth()}")
print(f"Number of Leaves: {model_scaled.get_n_leaves()}")

plt.figure(figsize=(20, 10))
plot_tree(model_scaled, feature_names=X_scaled.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()

def categorize_price(price):
    if price < train_data['Price'].quantile(0.25):
        return 'Low'
    elif price < train_data['Price'].quantile(0.5):
        return 'Medium'
    elif price < train_data['Price'].quantile(0.75):
        return 'High'
    else:
        return 'Very High'