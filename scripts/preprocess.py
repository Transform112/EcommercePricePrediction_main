import pandas as pd
import re

# Load the new dataset
df = pd.read_csv('./data/raw/ecommerce_dataset.csv')

# Preprocessing and mapping to schema
df['Category'] = df['Category'].str.capitalize()  # Normalize category names
df['SubCategory'] = df['Model']  # Use Model as SubCategory
df['Brand'] = df['Brand']

# Parse RAM (e.g., '8GB' -> 8, NA -> 0)
df['RAM (GB)'] = pd.to_numeric(df['RAM'].astype(str).str.extract(r'(\d+)')[0], errors='coerce').fillna(0)

# Parse Storage (e.g., '1TB HDD' -> 1024, '512GB SSD' -> 512, NA -> 0)
def parse_storage(storage_str):
    if pd.isna(storage_str):
        return 0
    match = re.match(r'(\d+)(\s*(TB|GB))?(\s*(HDD|SSD))?', str(storage_str), re.I)
    if match:
        value = int(match.group(1))
        unit = match.group(3).upper() if match.group(3) else 'GB'
        return value * 1024 if unit == 'TB' else value
    return 0
df['Storage (GB)'] = df['Storage'].apply(parse_storage)

df['CPU/Processor'] = df['Processor'].fillna('NA')
df['GPU'] = 'NA'  # No GPU info
df['Screen Size'] = pd.to_numeric(df['ScreenSize'], errors='coerce').fillna(0)
df['Battery (mAh)'] = pd.to_numeric(df['Battery'], errors='coerce').fillna(0)
df['Camera (MP)'] = 0  # No camera info
df['Capacity (Liters)'] = 0  # For appliances/furniture, but no specific; set to 0
df['Energy Rating'] = 'NA'  # No energy rating
df['Weight (Kg)'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0)
df['Material'] = df['Material'].fillna('NA')
df['Features'] = df.apply(lambda row: f"Size: {row['Size']}, Color: {row['Color']}, Dimensions: {row['Dimensions']}, Power: {row['Power']}, Warranty: {row['Warranty']}", axis=1)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)

# Unified schema
unified_df = df[['Category', 'SubCategory', 'Brand', 'RAM (GB)', 'Storage (GB)', 'CPU/Processor', 'GPU', 'Screen Size', 'Battery (mAh)', 'Camera (MP)', 'Capacity (Liters)', 'Energy Rating', 'Weight (Kg)', 'Material', 'Features', 'Price']]

# Clean: Filter relevant categories, handle types
relevant_categories = ['Laptop', 'Mobile', 'Appliance', 'Shoes', 'Furniture', 'Clothes']
unified_df = unified_df[unified_df['Category'].isin(relevant_categories)]

num_cols = ['RAM (GB)', 'Storage (GB)', 'Screen Size', 'Battery (mAh)', 'Camera (MP)', 'Capacity (Liters)', 'Weight (Kg)', 'Price']
for col in num_cols:
    unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce').fillna(0)

cat_cols = ['Brand', 'CPU/Processor', 'GPU', 'Energy Rating', 'Material', 'Features']
for col in cat_cols:
    unified_df[col] = unified_df[col].fillna('NA')

unified_df = unified_df.dropna(subset=['Price'])

# Save unified dataset
unified_df.to_csv('./data/processed/unified_ecommerce_dataset.csv', index=False)
print("Unified dataset saved. Shape:", unified_df.shape)
print(unified_df.head())