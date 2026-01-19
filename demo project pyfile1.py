import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Load CSV
# -------------------------------
csv_path = r"C:\Users\ulaga\Downloads\products.csv"   # <- adjust if needed
print("\nLoading:", csv_path)
df = pd.read_csv(csv_path, encoding='latin1', dtype=str)
print("Raw shape:", df.shape)
print("Columns (first 40 shown):", list(df.columns)[:40])

# -------------------------------
# Detect item column
# -------------------------------
def is_number_col(series):
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False
    parsed = non_null.apply(lambda x: bool(re.search(r'\d', x)) and bool(re.sub(r'[,\s₹\-–—\.]','',x).lstrip('-').isdigit()))
    return parsed.mean() > 0.6

item_col = None
for col in df.columns:
    s = df[col].astype(str).str.strip()
    alpha_frac = (s.str.contains(r'[A-Za-z]', na=False)).mean()
    if alpha_frac > 0.4 and not is_number_col(s):
        item_col = col
        break
if item_col is None:
    item_col = df.columns[0]

print("Detected item column:", item_col)

# -------------------------------
# Detect numeric cost columns
# -------------------------------
cols_lower = {c.lower(): c for c in df.columns}
min_col = max_col = avg_col = None
for key in cols_lower:
    if 'min' == key or key.startswith('min'):
        min_col = cols_lower[key]
    if 'max' == key or key.startswith('max'):
        max_col = cols_lower[key]
    if 'avg' == key or key.startswith('avg') or 'average' in key:
        avg_col = cols_lower[key]

def to_numeric_series(col):
    return pd.to_numeric(df[col].astype(str).str.replace(r'[₹,]', '', regex=True).str.strip(), errors='coerce')

parsed = pd.DataFrame()
parsed[item_col] = df[item_col].astype(str).fillna('').str.strip()

if min_col and max_col:
    parsed['Min'] = to_numeric_series(min_col)
    parsed['Max'] = to_numeric_series(max_col)
    parsed['Avg'] = to_numeric_series(avg_col) if avg_col else (parsed['Min'] + parsed['Max']) / 2
else:
    numeric_candidates = []
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        frac = s.apply(lambda x: bool(re.match(r'^[\d\-,\.\s₹–—]+$', x.strip()))).mean()
        if frac > 0.6:
            numeric_candidates.append(col)
    if len(numeric_candidates) >= 2:
        parsed['Min'] = to_numeric_series(numeric_candidates[0])
        parsed['Max'] = to_numeric_series(numeric_candidates[1])
        parsed['Avg'] = to_numeric_series(numeric_candidates[2]) if len(numeric_candidates) > 2 else (parsed['Min'] + parsed['Max']) / 2
    else:
        cost_col = [c for c in df.columns if c != item_col][0]
        raw = df[cost_col].astype(str).fillna('').str.strip()
        def parse_cost_text(txt):
            t = re.sub(r'[₹\s]', '', txt).replace('–', '-').replace('—', '-')
            parts = re.split(r'[-–—toTO]+', t)
            parts = [p.strip().replace(',', '') for p in parts if p.strip()]
            if len(parts) >= 2:
                try: return float(parts[0]), float(parts[1])
                except: return np.nan, np.nan
            elif parts:
                try: v = float(parts[0]); return v, v
                except: return np.nan, np.nan
            else: return np.nan, np.nan
        mins, maxs = zip(*(parse_cost_text(x) for x in raw))
        parsed['Min'] = pd.Series(mins)
        parsed['Max'] = pd.Series(maxs)
        parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2

# Ensure numeric
parsed['Min'] = pd.to_numeric(parsed['Min'], errors='coerce')
parsed['Max'] = pd.to_numeric(parsed['Max'], errors='coerce')
parsed['Avg'] = pd.to_numeric(parsed['Avg'], errors='coerce')
if parsed['Avg'].isna().all() and not (parsed['Min'].isna().all() and parsed['Max'].isna().all()):
    parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2

print("\nParsed sample (first 10 rows):")
print(parsed.head(10))

# -------------------------------
# Electronics filter
# -------------------------------
electronics_keywords = ["electronic", "electronics", "tv", "computer", "laptop", "mobile", "device"]
mask = parsed[item_col].str.lower().str.contains('|'.join(electronics_keywords))
electronics_df = parsed[mask].copy()

# -------------------------------
# Totals function
# -------------------------------
def calculate_totals(df, title="All Items"):
    valid = df.dropna(subset=['Avg']).copy()
    total_min = valid['Min'].sum()
    total_avg = valid['Avg'].sum()
    total_max = valid['Max'].sum()
    buffer_pct = ((total_max - total_min) / total_avg * 100) if total_avg else np.nan
    print(f"\n--- {title} ---")
    print(f"Total Min: {total_min:,.2f}")
    print(f"Total Avg: {total_avg:,.2f}")
    print(f"Total Max: {total_max:,.2f}")
    print("Budget buffer (%):", "N/A" if np.isnan(buffer_pct) else f"{buffer_pct:.1f}%")
    return valid

valid_all = calculate_totals(parsed, "All Items")
valid_elec = calculate_totals(electronics_df, "Electronics Only")

# -------------------------------
# Combined figure: All Items + Electronics
# -------------------------------
plt.style.use('seaborn-v0_8')
blue_shades = ["#b4561f", "#582EC1", "#3C34DB", '#5DADE2', "#E9E285", "#8C9296"]
fig, axs = plt.subplots(4, 2, figsize=(18, 20))  # 4 rows x 2 columns

datasets = [
    ("All Items", valid_all),
    ("Electronics Only", valid_elec)
]

for row_idx, (title, data) in enumerate(datasets):
    # Chart 1: Avg cost bar
    axs[row_idx*2, 0].bar(data[item_col], data['Avg'], color=blue_shades[2])
    axs[row_idx*2, 0].set_title(f"{title}: Cost Breakdown (Avg)")
    axs[row_idx*2, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(data['Avg']):
        axs[row_idx*2, 0].text(i, v + (abs(v)*0.02 if not np.isnan(v) else 0), f"{v:,.0f}", ha='center', fontsize=8)

    # Chart 2: Pie chart
    total_avg = data['Avg'].sum()
    shares = (data['Avg'] / total_avg * 100) if total_avg else np.zeros(len(data))
    axs[row_idx*2, 1].pie(shares, labels=data[item_col], autopct="%1.1f%%", colors=blue_shades)
    axs[row_idx*2, 1].set_title(f"{title}: Cost Distribution (%)")

    # Chart 3: Cost uncertainty
    ranges = (data['Max'] - data['Min']).fillna(0)
    axs[row_idx*2+1, 0].bar(data[item_col], ranges, color=blue_shades[4])
    axs[row_idx*2+1, 0].set_title(f"{title}: Cost Uncertainty (Max-Min)")
    axs[row_idx*2+1, 0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(ranges):
        axs[row_idx*2+1, 0].text(i, v + (abs(v)*0.02 if v!=0 else 0), f"{v:,.0f}", ha='center', fontsize=8)

    # Chart 4: Top 4 Avg costs
    top4 = data.nlargest(4, 'Avg')
    axs[row_idx*2+1, 1].bar(top4[item_col], top4['Avg'], color=blue_shades[:3])
    axs[row_idx*2+1, 1].set_title(f"{title}: Top 4 Highest Avg Costs")
    for i, v in enumerate(top4['Avg']):
        axs[row_idx*2+1, 1].text(i, v + (abs(v)*0.02 if not np.isnan(v) else 0), f"{v:,.0f}", ha='center', fontsize=7)

plt.suptitle("All Items vs Electronics Cost Analysis", fontsize=20)
plt.tight_layout(rect=(0, 0.03, 1, 0.97))
plt.show()


