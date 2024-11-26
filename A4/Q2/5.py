import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv('../dataset/train.csv')

# Plot the distribution of the 'Price' variable
plt.figure(figsize=(10, 6))
plt.hist(train_data['Price'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Property Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Define price brackets
def categorize_price(price):
    if price < 10000000:
        return 'Low'
    elif 10000000 <= price < 25000000:
        return 'Medium'
    elif 25000000 <= price < 50000000:
        return 'High'
    else:
        return 'Very High'

# Create a new column for price categories
train_data['Price_Category'] = train_data['Price'].apply(categorize_price)

# Display the first few rows to verify
print(train_data[['Price', 'Price_Category']].head())

# Count the distribution of price categories
category_counts = train_data['Price_Category'].value_counts()

# Plot the distribution of price categories
plt.figure(figsize=(8, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Distribution of Properties Across Price Categories')
plt.xlabel('Price Category')
plt.ylabel('Number of Properties')
plt.show()

# Print the counts for further analysis
print("\nDistribution of Properties Across Price Categories:")
print(category_counts)