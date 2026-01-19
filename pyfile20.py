import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ==================================================
# 1. CREATE 30 YEARS OF DATA (1995–2024)
# ==================================================
np.random.seed(10)

years = np.arange(1995, 2025)

df = pd.DataFrame({
    "Year": years,
    "Toyota": np.random.randint(15000, 50000, 30),
    "Honda": np.random.randint(12000, 48000, 30),
    "Hyundai": np.random.randint(13000, 52000, 30),
    "Maruti": np.random.randint(20000, 65000, 30),
    "Tata": np.random.randint(10000, 45000, 30),
    "Mahindra": np.random.randint(9000, 42000, 30)
})

print("\n 30 Years Car Sales Data:\n")
print(df)

# ==================================================
# 2. TRAIN 6 SEPARATE ML MODELS
# ==================================================
X = df[["Year"]]
future_years = np.arange(2025, 2031).reshape(-1, 1)

predictions = {"Year": future_years.flatten()}

models = {}

for car in df.columns[1:]:
    y = df[car]
    model = LinearRegression()
    model.fit(X, y)
    models[car] = model
    predictions[car] = model.predict(future_years).astype(int)

# ==================================================
# 3. PREDICT NEXT 6 YEARS
# ==================================================
future_df = pd.DataFrame(predictions)

print("\n Predicted Sales for Next 6 Years:\n")
print(future_df)

# ==================================================
# 4. ACTUAL vs PREDICTED GRAPH
# ==================================================
plt.figure(figsize=(12, 7))

for car in df.columns[1:]:
    plt.plot(df["Year"], df[car], label=f"{car} Actual")
    plt.plot(future_df["Year"], future_df[car], linestyle="--", label=f"{car} Predicted")

plt.xlabel("Year")
plt.ylabel("Sales Units")
plt.title("6 Car Models – Year-wise Sales Prediction")
plt.legend()
plt.grid(True)
plt.show()
