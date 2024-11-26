import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
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
X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])  # Drop non-predictive columns
y = train_data['Price_Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display initial distribution
print("Original Training Set Distribution:")
print(y_train.value_counts())

# Initialize Random Undersampler
undersampler = RandomUnderSampler(random_state=42)

# Apply undersampling
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Display distribution after undersampling
print("\nDistribution After Random Undersampling:")
print(y_train_under.value_counts())

# Initialize Random Oversampler
oversampler = RandomOverSampler(random_state=42)

# Apply oversampling
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

# Display distribution after oversampling
print("\nDistribution After Random Oversampling:")
print(y_train_over.value_counts())

# Visualize distributions
plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_under.value_counts().index, y=y_train_under.value_counts().values, palette='viridis')
plt.title('Distribution After Random Undersampling')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_over.value_counts().index, y=y_train_over.value_counts().values, palette='viridis')
plt.title('Distribution After Random Oversampling')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
plt.show()