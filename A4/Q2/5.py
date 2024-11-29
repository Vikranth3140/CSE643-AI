import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("Plots", exist_ok=True)
train_data = pd.read_csv('../dataset/train.csv')

plt.figure(figsize=(10, 6))
plt.hist(train_data['Price'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y')

output_path = "Plots/price_distribution.png"
plt.savefig(output_path, bbox_inches='tight')

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

train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

print(train_data[['Price', 'Price_Category']].head())

category_counts = train_data['Price_Category'].value_counts()

# Plot the distribution of price categories
plt.figure(figsize=(8, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Distribution of Properties Across Price Categories')
plt.xlabel('Price Category')
plt.ylabel('Number of Properties')

output_path = "Plots/price_category_distribution.png"
plt.savefig(output_path, bbox_inches='tight')

plt.show()

print("\nDistribution of Properties Across Price Categories:")
print(category_counts)