# --------------------------------------------
# 0. Imports
# --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
DATA_PATH = r"C:\Users\ulaga\Downloads\archive (1)\amazon.csv"

df = pd.read_csv(DATA_PATH)
print("\nDataset Loaded Successfully\n")
print(df.head())


# --------------------------------------------
# 2. Show Column Names (IMPORTANT)
# --------------------------------------------
print("\nColumn Names in Dataset:\n")
print(df.columns)


# --------------------------------------------
# 3. Remove Missing Values
# --------------------------------------------
df = df.dropna()


# --------------------------------------------
# 4. Encode Categorical Columns
# --------------------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])


# --------------------------------------------
# 5. Automatically Select Target Column
# --------------------------------------------
# Last numeric column will be used as target
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

target_col = numeric_cols[-1]   # SAFE AUTO SELECTION
print(f"\nTarget Column Selected: {target_col}\n")

X = df.drop(target_col, axis=1)
y = df[target_col]


# --------------------------------------------
# 6. Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------------------
# 7. Train Model
# --------------------------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Training started...\n")
model.fit(X_train, y_train)
print("Training completed\n")


# --------------------------------------------
# 8. Evaluation
# --------------------------------------------
y_pred = model.predict(X_test)

print("Model Performance")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))


# --------------------------------------------
# 9. Save Model
# --------------------------------------------
joblib.dump(model, "amazon_model.pkl")
print("\nModel saved as amazon_model.pkl\n")


# --------------------------------------------
# 10. Visualization
# --------------------------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Amazon Sales Prediction")
plt.show()
