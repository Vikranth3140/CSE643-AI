from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
train_data = pd.read_csv('../dataset/train.csv')

# Ensure 'Price_Category' column exists
def categorize_price(price):
    if price < train_data['Price'].quantile(0.25):
        return 'Low'
    elif price < train_data['Price'].quantile(0.5):
        return 'Medium'
    elif price < train_data['Price'].quantile(0.75):
        return 'High'
    else:
        return 'Very High'

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

# Prepare features (X) and target (y)
X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])
y = train_data['Price_Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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