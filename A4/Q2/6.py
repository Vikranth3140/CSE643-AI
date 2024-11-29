import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

train_data = pd.read_csv('../dataset/train.csv')

def categorize_price(price):
    if price < 10000000:
        return 'Low'
    elif 10000000 <= price < 25000000:
        return 'Medium'
    elif 25000000 <= price < 50000000:
        return 'High'
    else:
        return 'Very High'

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

X = train_data.drop(columns=['Price', 'Price_Category', 'Address', 'Possesion', 'Furnishing'])
y = train_data['Price_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original Training Set Distribution:")
print(y_train.value_counts())

# Apply undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Save undersampled data to CSV
undersampled_data = pd.concat([X_train_under, y_train_under], axis=1)
undersampled_data.to_csv('undersampled_data.csv', index=False)

print("\nDistribution After Random Undersampling:")
print(y_train_under.value_counts())

# Apply oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = oversampler.fit_resample(X_train, y_train)

# Save oversampled data to CSV
oversampled_data = pd.concat([X_train_over, y_train_over], axis=1)
oversampled_data.to_csv('oversampled_data.csv', index=False)

print("\nDistribution After Random Oversampling:")
print(y_train_over.value_counts())

plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_under.value_counts().index, y=y_train_under.value_counts().values, palette='viridis')
plt.title('Distribution After Random Undersampling')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
output_path = "Plots/undersampling_price_category_distribution.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=y_train_over.value_counts().index, y=y_train_over.value_counts().values, palette='viridis')
plt.title('Distribution After Random Oversampling')
plt.xlabel('Price Category')
plt.ylabel('Number of Samples')
output_path = "Plots/oversampling_price_category_distribution.png"
plt.savefig(output_path, bbox_inches='tight')
plt.show()