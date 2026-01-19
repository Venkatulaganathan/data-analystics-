import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------
# Step 1: Create dataset
# -------------------------
data = {
    'Size': [800, 900, 1000, 1100, 1200],
    'Rooms': [2, 2, 3, 3, 4],
    'Age': [10, 8, 6, 5, 2],
    'Price': [200000, 230000, 260000, 290000, 320000]
}

df = pd.DataFrame(data)

# -------------------------
# Step 2: Features & Target
# -------------------------
X = df[['Size', 'Rooms', 'Age']]   # input features
y = df['Price']                    # output

# -------------------------
# Step 3: Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Step 4: Train model
# -------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Step 5: Predict new data
# -------------------------
new_data = [[1050, 3, 4]]   # Size, Rooms, Age
prediction = model.predict(new_data)

print("Predicted Price:", int(prediction[0]))
