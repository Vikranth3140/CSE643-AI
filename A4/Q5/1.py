import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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



# Display initial distribution
print("Original Training Set Distribution:")
print(y_train.value_counts())

# Initialize SMOTE and apply
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nSMOTE Distribution:")
print(y_train_smote.value_counts())

# Introduce artificial imbalance
undersampler = RandomUnderSampler(sampling_strategy={'Low': 100, 'Medium': 100, 'High': 100, 'Very High': 50})
X_train_imbalanced, y_train_imbalanced = undersampler.fit_resample(X_train, y_train)

print("\nImbalanced Training Set Distribution (Before ADASYN):")
print(y_train_imbalanced.value_counts())

# Apply ADASYN on imbalanced data
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_imbalanced, y_train_imbalanced)
print("\nADASYN Distribution:")
print(y_train_adasyn.value_counts())

# Visualize distributions
plt.figure(figsize=(12, 6))

# SMOTE
plt.subplot(1, 2, 1)
smote_counts = y_train_smote.value_counts()
plt.bar(smote_counts.index, smote_counts.values, color='skyblue', edgecolor='black')
plt.title("SMOTE Distribution")
plt.xlabel("Price Categories")
plt.ylabel("Number of Samples")

# ADASYN
plt.subplot(1, 2, 2)
adasyn_counts = y_train_adasyn.value_counts()
plt.bar(adasyn_counts.index, adasyn_counts.values, color='orange', edgecolor='black')
plt.title("ADASYN Distribution")
plt.xlabel("Price Categories")
plt.ylabel("Number of Samples")

plt.tight_layout()
plt.show()

# Train Decision Tree Classifier on SMOTE-balanced data
dt_smote = DecisionTreeClassifier(random_state=42)
dt_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = dt_smote.predict(X_test)

# Train Decision Tree Classifier on ADASYN-balanced data
dt_adasyn = DecisionTreeClassifier(random_state=42)
dt_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_adasyn = dt_adasyn.predict(X_test)

# Evaluate SMOTE model
print("\nSMOTE Model Performance:")
print(classification_report(y_test, y_pred_smote))
print(f"SMOTE Accuracy: {accuracy_score(y_test, y_pred_smote):.2f}\n")

# Evaluate ADASYN model
print("\nADASYN Model Performance:")
print(classification_report(y_test, y_pred_adasyn))
print(f"ADASYN Accuracy: {accuracy_score(y_test, y_pred_adasyn):.2f}")