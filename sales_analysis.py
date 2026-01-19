import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD FILES
# ===============================
sales = pd.read_csv(r"C:\Users\ulaga\Downloads\archive (4)\sales.csv")
products = pd.read_csv(r"C:\Users\ulaga\Downloads\archive (4)\products.csv")
category = pd.read_csv(r"C:\Users\ulaga\Downloads\archive (4)\category.csv")
stores = pd.read_csv(r"C:\Users\ulaga\Downloads\archive (4)\stores.csv")
warranty = pd.read_csv(r"C:\Users\ulaga\Downloads\archive (4)\warranty.csv")

# ===============================
# PRINT COLUMNS (DEBUG)
# ===============================
print("Sales:", sales.columns.tolist())
print("Products:", products.columns.tolist())
print("Category:", category.columns.tolist())
print("Stores:", stores.columns.tolist())
print("Warranty:", warranty.columns.tolist())

# ===============================
# RENAME COMMON KEYS (STANDARDIZE)
# ===============================
sales.rename(columns={'prod_id':'product_id','product_code':'product_id'}, inplace=True)
products.rename(columns={'id':'product_id','product_code':'product_id'}, inplace=True)
category.rename(columns={'id':'category_id'}, inplace=True)
stores.rename(columns={'id':'store_id'}, inplace=True)

# ===============================
# MERGING
# ===============================
df = sales.merge(products, on='product_id', how='left')
df = df.merge(category, on='category_id', how='left')
df = df.merge(stores, on='store_id', how='left')
df = df.merge(warranty, on='product_id', how='left')

print("\nMerged successfully ✅")
print(df.head())

# ===============================
# SALES CALCULATION
# ===============================
df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
df['total_sales'] = df['quantity'] * df['price']

print("\nTotal Revenue:", df['total_sales'].sum())

# ===============================
# CATEGORY SALES
# ===============================
cat_sales = df.groupby('category_name')['total_sales'].sum()
cat_sales.plot(kind='bar', title='Category Wise Sales')
plt.show()


