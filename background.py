import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys
import warnings
warnings.filterwarnings('ignore')

print("Bio-CNG Project Cost Analysis - BLUE THEME")
print("=" * 50)

# -----------------------------------------
# STEP 1: LOAD DATASET (CHANGE ONLY THIS)
# -----------------------------------------
csv_path = r"C:\Users\ulaga\Downloads\products.csv"
df = pd.read_csv(csv_path, encoding='latin1')


print("Dataset loaded successfully!")
print("Shape:", df.shape)

# Identify columns
item_col = df.columns[0]
cost_col = df.columns[1]

df[item_col] = df[item_col].astype(str)

# -----------------------------------------
# STEP 2: COST PARSER
# -----------------------------------------
def parse_cost(clean_cost):
    clean = re.sub(r'[₹,]', '', str(clean_cost))
    clean = re.sub(r'[–—-]', '-', clean)

    parts = re.split(r'[-–—]', clean)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) >= 2:
        try:
            return float(parts[0]) / 100000, float(parts[1]) / 100000
        except:
            return np.nan, np.nan
    else:
        try:
            v = float(parts[0])
            return v / 100000, v / 100000
        except:
            return np.nan, np.nan

df[['Min_Lakhs', 'Max_Lakhs']] = df[cost_col].apply(lambda x: pd.Series(parse_cost(x)))
df['Avg_Lakhs'] = (df['Min_Lakhs'] + df['Max_Lakhs']) / 2

print("\nParsed Costs (Lakhs):")
print(df[[item_col, 'Min_Lakhs', 'Max_Lakhs', 'Avg_Lakhs']].round(2))

# -----------------------------------------
# STEP 3: TOTAL COST
# -----------------------------------------
valid = df.dropna(subset=['Avg_Lakhs'])

total_min = valid['Min_Lakhs'].sum()
total_max = valid['Max_Lakhs'].sum()
total_avg = valid['Avg_Lakhs'].sum()

print("\nTOTAL PROJECT COST:")
print(f"Min: {round(total_min)} Lakhs")
print(f"Avg: {round(total_avg)} Lakhs")
print(f"Max: {round(total_max)} Lakhs")

# -----------------------------------------
# STEP 4: PLOTS (BLUE THEME)
# -----------------------------------------
plt.style.use('seaborn-v0_8')
blue_shades = ['#1f77b4', '#2E86C1', '#3498DB', '#5DADE2', '#85C1E9', '#AED6F1']

fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cost Breakdown
axs[0,0].barh(valid[item_col], valid['Avg_Lakhs'], color=blue_shades)
axs[0,0].set_title("Cost Breakdown (Lakhs)", color='blue', fontsize=14)
axs[0,0].set_xlabel("Lakhs")
axs[0,0].invert_yaxis()

# 2. Pie Chart
axs[0,1].pie(
    valid['Avg_Lakhs'],
    labels=[x[:12] + ".." for x in valid[item_col]],
    autopct='%1.1f%%',
    startangle=90,
    colors=blue_shades
)
axs[0,1].set_title("Cost Distribution (%)", color='blue', fontsize=14)

# 3. Cost Range
ranges = valid['Max_Lakhs'] - valid['Min_Lakhs']
axs[1,0].bar(range(len(ranges)), ranges, color=blue_shades[2])
axs[1,0].set_xticks(range(len(valid)))
axs[1,0].set_xticklabels([x[:10] for x in valid[item_col]], rotation=45)
axs[1,0].set_title("Cost Uncertainty (Lakhs)", color='blue', fontsize=14)

# 4. Top 3 Highest Cost Items
top3 = valid.nlargest(3, 'Avg_Lakhs')
axs[1,1].barh(top3[item_col], top3['Avg_Lakhs'], color=blue_shades[:3])
axs[1,1].set_title("Top 3 Highest Costs", color='blue', fontsize=14)
axs[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig("bio_cng_blue_analysis.png", dpi=300)
plt.show()

# -----------------------------------------
# STEP 5: TEXT SUMMARY
# -----------------------------------------
print("\nBLUE THEME INSIGHTS:")
print(f"Total Avg Investment: Rs {int(total_avg * 100000):,}")
print(f"Budget Buffer: {round(((total_max - total_min) / total_avg) * 100)}%")

print("\nTop 3 Cost Items:")
for i, (idx, row) in enumerate(top3.iterrows(), 1):
    print(f"{i}. {row[item_col]} — Rs {int(row['Avg_Lakhs']*100000):,}")

print("\nRECOMMENDATIONS:")
print("- Digester + Civil work forms major cost — negotiate wisely.")
print("- Apply for MNRE Subsidy early.")
print("- Compare vendor prices before finalizing equipment.")

print("\nPROGRAM COMPLETED SUCCESSFULLY!")
