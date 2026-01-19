import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv")

X = df[['Size', 'Rooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

size = int(input("Enter house size: "))
rooms = int(input("Enter number of rooms: "))
age = int(input("Enter house age: "))

prediction = model.predict([[size, rooms, age]])
print("Predicted House Price:", int(prediction[0]))
