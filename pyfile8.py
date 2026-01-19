import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# ============================
# Updated Path Here
# ============================
csv_path = r"C:\Users\ulaga\Downloads\py files\py files\transactions.csv"

print("\nLoading:", csv_path)
df = pd.read_csv(csv_path, encoding='latin1', dtype=str)  # load as strings to inspect
print("Raw shape:", df.shape)
print("Columns (first 40 shown):", list(df.columns)[:40])

# --- Helper: detect item column (first non-numeric / object-like column) ---
def is_number_col(series):
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False
    parsed = non_null.apply(lambda x: bool(re.search(r'\d', x)) and bool(re.sub(r'[,\s₹₹₹\-–—\.]','',x).lstrip('-').isdigit()))
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

# --- Detect cost columns ---
cols_lower = {c.lower(): c for c in df.columns}
min_col = None; max_col = None; avg_col = None

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
    print("Found explicit Min/Max columns:", min_col, max_col, " (Avg found:" , avg_col, ")")
    parsed['Min'] = to_numeric_series(min_col)
    parsed['Max'] = to_numeric_series(max_col)
    if avg_col:
        parsed['Avg'] = to_numeric_series(avg_col)
    else:
        parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2
else:
    numeric_candidates = []
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        frac = s.apply(lambda x: bool(re.match(r'^[\d\-,\.\s₹₹₹–—]+$', x.strip()))).mean()
        if frac > 0.6:
            numeric_candidates.append(col)

    if len(numeric_candidates) >= 2:
        print("Inferred numeric columns for costs:", numeric_candidates[:3])
        parsed['Min'] = to_numeric_series(numeric_candidates[0])
        parsed['Max'] = to_numeric_series(numeric_candidates[1])
        if len(numeric_candidates) > 2:
            parsed['Avg'] = to_numeric_series(numeric_candidates[2])
        else:
            parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2
    else:
        cost_col = [c for c in df.columns if c != item_col][0]
        print("No explicit numeric columns found; parsing single cost/text column:", cost_col)
        raw = df[cost_col].astype(str).fillna('').str.strip()

        def parse_cost_text(txt):
            t = re.sub(r'[₹\s]', '', txt)
            t = t.replace('–', '-').replace('—', '-')
            parts = re.split(r'[-–—toTO]+', t)
            parts = [p.strip().replace(',', '') for p in parts if p.strip()]
            if len(parts) >= 2:
                try:
                    return float(parts[0]), float(parts[1])
                except:
                    return np.nan, np.nan
            else:
                try:
                    v = float(parts[0].replace(',', ''))
                    return v, v
                except:
                    return np.nan, np.nan

        mins = []; maxs = []
        for x in raw:
            a,b = parse_cost_text(x)
            mins.append(a); maxs.append(b)
        parsed['Min'] = pd.Series(mins)
        parsed['Max'] = pd.Series(maxs)
        parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2

def normalize_units(s):
    s = pd.to_numeric(s, errors='coerce')
    return s

parsed['Min'] = normalize_units(parsed['Min'])
parsed['Max'] = normalize_units(parsed['Max'])
parsed['Avg'] = normalize_units(parsed['Avg'])

if parsed['Avg'].isna().all() and not (parsed['Min'].isna().all() and parsed['Max'].isna().all()):
    parsed['Avg'] = (parsed['Min'] + parsed['Max']) / 2

print("\nParsed sample (first 10 rows):")
print(parsed.head(10))
print("Parsed shape:", parsed.shape)

valid = parsed.dropna(subset=['Avg']).copy()
if valid.shape[0] == 0:
    raise ValueError("No valid numeric cost rows were parsed. Please check CSV contents or send sample rows here.")

total_min = valid['Min'].sum()
total_avg = valid['Avg'].sum()
total_max = valid['Max'].sum()
print("\nTotals:")
print(f"Total Min: {total_min:,.2f}")
print(f"Total Avg: {total_avg:,.2f}")
print(f"Total Max: {total_max:,.2f}")

if total_avg and not np.isnan(total_avg) and total_avg != 0:
    buffer_pct = ((total_max - total_min) / total_avg) * 100
else:
    buffer_pct = np.nan

print("Budget buffer (%) :", "N/A" if np.isnan(buffer_pct) else f"{buffer_pct:.1f}%")

plt.style.use('seaborn-v0_8')
blue_shades = ["#b4561f", "#582EC1", "#3C34DB", '#5DADE2', "#E9E285", "#8C9296"]

import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(16, 11))

# Chart 1
axs[0,0].bar(valid[item_col], valid['Avg'], color=blue_shades[2])
axs[0,0].set_title("Cost Breakdown (Avg)")
axs[0,0].set_ylabel("Cost (units)")
axs[0,0].tick_params(axis='x', rotation=45)

for i, v in enumerate(valid['Avg']):
    axs[0,0].text(i, v + (abs(v)*0.02 if not np.isnan(v) else 0), f"{v:,.0f}", ha='center', fontsize=9)

# Chart 2
shares = (valid['Avg'] / total_avg) * 100 if total_avg else np.zeros(len(valid))
axs[0,1].pie(shares, labels=valid[item_col], autopct="%1.1f%%", colors=blue_shades)
axs[0,1].set_title("Cost Distribution (%)")

# Chart 3
ranges = (valid['Max'] - valid['Min']).fillna(0)
axs[1,0].bar(valid[item_col], ranges, color=blue_shades[4])
axs[1,0].set_title("Cost Uncertainty (Max - Min)")
axs[1,0].tick_params(axis='x', rotation=45)

for i, v in enumerate(ranges):
    axs[1,0].text(i, v + (abs(v)*0.02 if v!=0 else 0), f"{v:,.0f}", ha='center', fontsize=9)

# Chart 4
top3 = valid.nlargest(3, 'Avg')
axs[1,1].bar(top3[item_col], top3['Avg'], color=blue_shades[:3])
axs[1,1].set_title("Top 3 Highest Costs")

for i, v in enumerate(top3['Avg']):
    axs[1,1].text(i, v + (abs(v)*0.02 if not np.isnan(v) else 0), f"{v:,.0f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("bio_cng_auto_parsed.png", dpi=300)
plt.show(block=True)

print("\nSaved plot: bio_cng_auto_parsed.png")
