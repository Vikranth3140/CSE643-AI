import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Identify categorical columns
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

# Identify numerical columns for scaling
numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns

# Initialize StandardScaler
scaler = StandardScaler()

# Scale the numerical features
scaled_features = scaler.fit_transform(train_data[numerical_columns])

# Replace the original numerical data with scaled data
train_data[numerical_columns] = scaled_features

# Display a preview of scaled features
print("\nScaled Numerical Features:")
print(train_data[numerical_columns].head())


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Split the original (unscaled) dataset
X_unscaled = train_data.drop(columns=['Price'])  # Replace 'Price' with the actual target column
y_unscaled = train_data['Price']
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(X_unscaled, y_unscaled, test_size=0.2, random_state=42)

# Train Decision Tree on unscaled data
model_unscaled = DecisionTreeRegressor(random_state=42)
model_unscaled.fit(X_train_unscaled, y_train_unscaled)

# Predict and calculate performance
y_pred_unscaled = model_unscaled.predict(X_test_unscaled)
mse_unscaled = mean_squared_error(y_test_unscaled, y_pred_unscaled)
print(f"\nMean Squared Error (Unscaled Data): {mse_unscaled}")

# Split the scaled dataset
X_scaled = train_data.drop(columns=['Price'])  # Replace 'Price' with the actual target column
y_scaled = train_data['Price']
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train Decision Tree on scaled data
model_scaled = DecisionTreeRegressor(random_state=42)
model_scaled.fit(X_train_scaled, y_train_scaled)

# Predict and calculate performance
y_pred_scaled = model_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error (Scaled Data): {mse_scaled}")