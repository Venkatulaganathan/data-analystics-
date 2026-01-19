# CUSTOMER PURCHASE BEHAVIOR ANALYSIS
# Fully Working Program for marketing_campaign.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------
csv_path = r"C:\Users\ulaga\Downloads\py files\py files\marketing_campaign.csv"

try:
    df = pd.read_csv(csv_path, sep="\t")  # Tab-separated CSV
    print("Dataset loaded successfully")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

print("Rows:", len(df))
print("Columns:", list(df.columns))

# ---------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------
df = df.dropna(subset=["Income"])
df["Income"] = df["Income"].fillna(df["Income"].median())
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")

# ---------------------------------------------------
# 3. Create RFM Features
# ---------------------------------------------------
df["Recency"] = df["Recency"]
df["Frequency"] = (
    df["NumDealsPurchases"]
    + df["NumWebPurchases"]
    + df["NumCatalogPurchases"]
    + df["NumStorePurchases"]
)
df["Monetary"] = (
    df["MntWines"] 
    + df["MntFruits"] 
    + df["MntMeatProducts"] 
    + df["MntFishProducts"] 
    + df["MntSweetProducts"] 
    + df["MntGoldProds"]
)
print("\nRFM Features Created")
print(df[["Recency", "Frequency", "Monetary"]].head())

# ---------------------------------------------------
# 4. K-Means Clustering
# ---------------------------------------------------
feature_cols = ["Recency", "Frequency", "Monetary"]
X = df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nClustering Completed. Cluster distribution:")
print(df["Cluster"].value_counts())

# ---------------------------------------------------
# 5. Visualizations: 4 plots in 1 figure
# ---------------------------------------------------
plt.style.use('seaborn-v0_8')
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 1. Frequency vs Monetary
scatter1 = axs[0,0].scatter(df["Frequency"], df["Monetary"], c=df["Cluster"], s=25, cmap='tab10')
axs[0,0].set_xlabel("Frequency")
axs[0,0].set_ylabel("Monetary")
axs[0,0].set_title("Frequency vs Monetary")

# 2. Recency vs Monetary
scatter2 = axs[0,1].scatter(df["Recency"], df["Monetary"], c=df["Cluster"], s=25, cmap='tab10')
axs[0,1].set_xlabel("Recency")
axs[0,1].set_ylabel("Monetary")
axs[0,1].set_title("Recency vs Monetary")

# 3. Recency vs Frequency
scatter3 = axs[1,0].scatter(df["Recency"], df["Frequency"], c=df["Cluster"], s=25, cmap='tab10')
axs[1,0].set_xlabel("Recency")
axs[1,0].set_ylabel("Frequency")
axs[1,0].set_title("Recency vs Frequency")

# 4. Monetary vs Income
scatter4 = axs[1,1].scatter(df["Income"], df["Monetary"], c=df["Cluster"], s=25, cmap='tab10')
axs[1,1].set_xlabel("Income")
axs[1,1].set_ylabel("Monetary")
axs[1,1].set_title("Monetary vs Income")

# Common legend
handles, labels = scatter1.legend_elements()
fig.legend(handles, [f"Cluster {i}" for i in range(len(df['Cluster'].unique()))], loc="upper right", fontsize=10)

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# 6. Segment Interpretation
# ---------------------------------------------------
segment_summary = df.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
print("\nCluster Summary (Higher is better except Recency):")
print(segment_summary)

print("\nProgram completed successfully.")
