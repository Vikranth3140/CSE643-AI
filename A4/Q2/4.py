import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# train_data = pd.read_csv('../dataset/train.csv')
# test_data = pd.read_csv('../dataset/test.csv')
train_data = pd.read_csv('dropped_cols_train_data.csv')
test_data = pd.read_csv('dropped_cols_test_data.csv')

# First we encode the categorical columns and then scale the numerical columns

# Here, we assume that all the columns with 'object' data type are categorical columns, including "Address"
categorical_columns = train_data.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Label encoding the categorical columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    train_data[col] = label_encoder.fit_transform(train_data[col])
    test_data[col] = label_encoder.fit_transform(test_data[col])
    print(f"Column '{col}' encoded with Label Encoding.")

numerical_columns = train_data.select_dtypes(include=['float64', 'int64']).columns


# I decided to do the analysis here itself on training Decision Tree on scaled and unscaled data

# Unscaled data
X_unscaled = train_data.drop(columns=['Price'])
y_unscaled = train_data['Price']

X_unscaled_test = test_data.drop(columns=['Price'])
y_unscaled_test = test_data['Price']

# Train Decision Tree on unscaled data
model_unscaled = DecisionTreeRegressor(random_state=42)
model_unscaled.fit(X_unscaled, y_unscaled)
y_pred_unscaled = model_unscaled.predict(X_unscaled_test)

mse_unscaled = mean_squared_error(y_unscaled_test, y_pred_unscaled)
print(f"\nMean Squared Error (Unscaled Data): {mse_unscaled}")

r2_unscaled = r2_score(y_unscaled_test, y_pred_unscaled)
print(f"R² Score (Unscaled Data): {r2_unscaled}")

mae_unscaled = mean_absolute_error(y_unscaled_test, y_pred_unscaled)
print(f"Mean Absolute Error (Unscaled Data): {mae_unscaled}")



# Scaled data
columns_to_exclude = ['Price', 'Address', 'Possesion', 'Furnishing']
numerical_columns_to_scale = [col for col in train_data.select_dtypes(include=['float64', 'int64']).columns if col not in columns_to_exclude]

y_train = train_data['Price']
y_test = test_data['Price']

# Apply StandardScaler to scale the features
scaler = StandardScaler()

X_train_scaled_numerical = scaler.fit_transform(train_data[numerical_columns_to_scale])
X_test_scaled_numerical = scaler.transform(test_data[numerical_columns_to_scale])

X_train_scaled_numerical_df = pd.DataFrame(X_train_scaled_numerical, columns=numerical_columns_to_scale)
X_test_scaled_numerical_df = pd.DataFrame(X_test_scaled_numerical, columns=numerical_columns_to_scale)

excluded_columns_train = train_data[columns_to_exclude].reset_index(drop=True)
excluded_columns_test = test_data[columns_to_exclude].reset_index(drop=True)

X_train_final = pd.concat([X_train_scaled_numerical_df, excluded_columns_train], axis=1)
X_test_final = pd.concat([X_test_scaled_numerical_df, excluded_columns_test], axis=1)

# Train Decision Tree on scaled data
model_scaled = DecisionTreeRegressor(random_state=42)
model_scaled.fit(X_train_final, y_train)
y_pred_scaled = model_scaled.predict(X_test_final)

mse_scaled = mean_squared_error(y_test, y_pred_scaled)
print(f"\nMean Squared Error (Scaled Data): {mse_scaled}")

r2_scaled = r2_score(y_test, y_pred_scaled)
print(f"R² Score (Scaled Data): {r2_scaled}")

mae_scaled = mean_absolute_error(y_test, y_pred_scaled)
print(f"Mean Absolute Error (Scaled Data): {mae_scaled}")