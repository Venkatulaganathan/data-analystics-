import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Step 1: Load CSV safely
# -----------------------------
csv_path = r"C:\Users\ulaga\Downloads\py files\py files\Instagram_Analytics.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Fix single-string header issue
if len(df.columns) == 1 and ',' in df.columns[0]:
    df = pd.read_csv(csv_path, sep=',')

print(f"Loaded rows: {len(df)}")
print(f"Columns detected: {list(df.columns)}")

# -----------------------------
# Step 2: Ensure required columns
# -----------------------------
required_cols = [
    'post_id','upload_date','media_type','likes','comments','shares','saves',
    'reach','impressions','caption_length','hashtags_count','followers_gained',
    'traffic_source','engagement_rate','content_category'
]

for col in required_cols:
    if col not in df.columns:
        if col == 'engagement_rate' and all(
            x in df.columns for x in ['likes','comments','shares','saves','followers_gained']
        ):
            df['engagement_rate'] = (
                df['likes'] + df['comments'] + df['shares'] + df['saves']
            ) / df['followers_gained'].replace(0, np.nan)
        else:
            df[col] = 0

# Ensure numeric types
numeric_fix = ['likes','comments','shares','saves','reach','impressions','followers_gained','engagement_rate']
for c in numeric_fix:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# -----------------------------
# Step 3: Save cleaned dataset
# -----------------------------
output_dir = "instagram_analysis_outputs"
os.makedirs(output_dir, exist_ok=True)
cleaned_csv_path = os.path.join(output_dir, "cleaned_instagram.csv")
df.to_csv(cleaned_csv_path, index=False)
print(f"Saved cleaned dataset: {cleaned_csv_path}")

# -----------------------------
# Step 4: Clustering (KMeans)
# -----------------------------
numeric_cols = ['likes','comments','shares','saves','reach','impressions']
available_numeric = [c for c in numeric_cols if c in df.columns]

if len(available_numeric) >= 2:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[available_numeric])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print("Clustering applied: 3 clusters created.")
else:
    df['Cluster'] = -1
    print("Not enough numeric features for clustering.")

# -----------------------------
# Step 5: PCA for visualization
# -----------------------------
df_pca = None
if len(available_numeric) >= 2:
    pca = PCA(n_components=2, random_state=42)
    df_pca = pca.fit_transform(df[available_numeric])

# -----------------------------
# Step 6: SINGLE DASHBOARD (2x2)
# -----------------------------
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Plot 1: Histogram of Likes
sns.histplot(data=df, x='likes', bins=30, kde=True, ax=axs[0, 0])
axs[0, 0].set_title("Histogram of Likes")

# Plot 2: PCA Cluster Scatter
if df_pca is not None and 'Cluster' in df.columns:
    sns.scatterplot(
        x=df_pca[:, 0],
        y=df_pca[:, 1],
        hue=df['Cluster'],
        palette='Set2',
        s=50,
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Cluster Visualization (PCA)")
    axs[0, 1].legend(title="Cluster")
else:
    axs[0, 1].text(0.5, 0.5, "Cluster data not available",ha='center', va='center', transform=axs[0, 1].transAxes)

# Plot 3: Average Likes per Cluster
if 'Cluster' in df.columns:
    avg_likes = df.groupby('Cluster')['likes'].mean().reset_index()
    sns.barplot(data=avg_likes, x='Cluster', y='likes', ax=axs[1, 0])
    axs[1, 0].set_title("Average Likes per Cluster")

# Plot 4: Engagement Rate Distribution
if 'engagement_rate' in df.columns:
    sns.boxplot(data=df, y='engagement_rate', ax=axs[1, 1])
    axs[1, 1].set_title("Engagement Rate Distribution")

# Final layout
plt.suptitle("Instagram Analytics Dashboard", fontsize=18)
plt.tight_layout(rect=(0, 0.03, 1, 0.95))

dashboard_path = os.path.join(output_dir, "instagram_dashboard.png")
plt.savefig(dashboard_path)
plt.show()

print(f"Saved dashboard: {dashboard_path}")

# -----------------------------
# Step 7: Cluster Summary
# -----------------------------
if 'Cluster' in df.columns:
    summary = df.groupby("Cluster").agg(
        posts=("post_id","count"),
        avg_likes=("likes","mean"),
        avg_comments=("comments","mean"),
        avg_shares=("shares","mean"),
        avg_saves=("saves","mean"),
        avg_impressions=("impressions","mean"),
        avg_engagement=("engagement_rate","mean")
    ).reset_index()

    print("\nCluster Summary:")
    print(summary)

    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved cluster summary: {summary_path}")
