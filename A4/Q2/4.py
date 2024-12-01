import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# train_data = pd.read_csv('../dataset/train.csv')
train_data = pd.read_csv('dropped_cols_train_data.csv')

# First we encode the categorical columns and then scale the numerical columns

# Here, we assume that all the columns with 'object' data type are categorical columns, including "Address"
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Label encoding the categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns


# I decided to do the analysis here itself on training Decision Tree on scaled and unscaled data

# Training Decision Tree on unscaled data
X_unscaled = train_data.drop(columns=['Price'])
y_unscaled = train_data['Price']
X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(X_unscaled, y_unscaled, test_size=0.2, random_state=42)

model_unscaled = DecisionTreeRegressor(random_state=42)
model_unscaled.fit(X_train_unscaled, y_train_unscaled)

y_pred_unscaled = model_unscaled.predict(X_test_unscaled)
mse_unscaled = mean_squared_error(y_test_unscaled, y_pred_unscaled)
print(f"\nMean Squared Error (Unscaled Data): {mse_unscaled}")


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

y_pred_scaled = model_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
print(f"Mean Squared Error (Scaled Data): {mse_scaled}")