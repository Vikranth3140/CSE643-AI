import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

os.makedirs("Plots", exist_ok=True)

train_data = pd.read_csv('../Q2/X_train_final_with_categories.csv')
test_data = pd.read_csv('../Q2/X_test_final_with_categories.csv')

X_train = train_data.drop(columns=['Price', 'Price_Category'])
y_train = train_data['Price_Category']

X_test = test_data.drop(columns=['Price', 'Price_Category'])
y_test = test_data['Price_Category']

if y_train.dtype == 'object':
    y_train = y_train.astype('category').cat.codes
    y_test = y_test.astype('category').cat.codes

# Display initial distribution
print("Original Training Set Distribution:")
print(y_train.value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nSMOTE Distribution:")
print(pd.Series(y_train_smote).value_counts())

# Introduce artificial imbalance
undersampler = RandomUnderSampler(sampling_strategy={0: 100, 1: 100, 2: 100, 3: 50}, random_state=42)
X_train_imbalanced, y_train_imbalanced = undersampler.fit_resample(X_train, y_train)

print("\nImbalanced Training Set Distribution (Before ADASYN):")
print(pd.Series(y_train_imbalanced).value_counts())

# Apply ADASYN on imbalanced data
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_imbalanced, y_train_imbalanced)
print("\nADASYN Distribution:")
print(pd.Series(y_train_adasyn).value_counts())

# Visualize distributions
plt.figure(figsize=(12, 6))

# SMOTE
plt.subplot(1, 2, 1)
smote_counts = pd.Series(y_train_smote).value_counts()
plt.bar(smote_counts.index.astype(str), smote_counts.values, color='skyblue', edgecolor='black')
plt.title("SMOTE Distribution")
plt.xlabel("Price Categories")
plt.ylabel("Number of Samples")

# ADASYN
plt.subplot(1, 2, 2)
adasyn_counts = pd.Series(y_train_adasyn).value_counts()
plt.bar(adasyn_counts.index.astype(str), adasyn_counts.values, color='orange', edgecolor='black')
plt.title("ADASYN Distribution")
plt.xlabel("Price Categories")
plt.ylabel("Number of Samples")

plt.tight_layout()

output_path = "Plots/SMOTE_and_ADASYN_distribution.png"
plt.savefig(output_path, bbox_inches='tight')
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
print(classification_report(y_test, y_pred_smote, target_names=['Low', 'Medium', 'High', 'Very High']))
print(f"SMOTE Accuracy: {accuracy_score(y_test, y_pred_smote):.2f}\n")

# Evaluate ADASYN model
print("\nADASYN Model Performance:")
print(classification_report(y_test, y_pred_adasyn, target_names=['Low', 'Medium', 'High', 'Very High']))
print(f"ADASYN Accuracy: {accuracy_score(y_test, y_pred_adasyn):.2f}")