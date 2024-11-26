import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv('../dataset/train.csv')

# Define price brackets for target variable
def categorize_price(price):
    if price < 10000000:
        return 'Low'
    elif 10000000 <= price < 25000000:
        return 'Medium'
    elif 25000000 <= price < 50000000:
        return 'High'
    else:
        return 'Very High'

# Create Price_Category column
train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

# Prepare features (X) and target (y)
X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])  # Drop categorical and target columns
y = train_data['Price']  # Use Price for regression

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Model evaluation
print(f"Decision Tree Depth: {dt_regressor.get_depth()}")
print(f"Number of Leaves: {dt_regressor.get_n_leaves()}")

# Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(dt_regressor, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure")
plt.show()