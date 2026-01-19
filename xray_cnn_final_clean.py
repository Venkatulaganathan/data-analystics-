# ============================================
# Chest X-Ray Pneumonia Detection using CNN
# ============================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # reduce tensorflow logs
import tensorflow as tf     
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Dataset Paths (YOUR PATH)
# --------------------------------------------
BASE_DIR = r"C:\Users\ulaga\Downloads\archive (7)\chest_xray\chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# --------------------------------------------
# 2. Image Parameters
# --------------------------------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 5

# --------------------------------------------
# 3. Data Preprocessing
# --------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# --------------------------------------------
# 4. CNN Model Architecture
# --------------------------------------------
model = Sequential([
    Input(shape=(150, 150, 3)),

    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# --------------------------------------------
# 5. Compile Model
# --------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------
# 6. Train Model
# --------------------------------------------
print("\nTraining started...\n")

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

# --------------------------------------------
# 7. Save Model
# --------------------------------------------
model.save("xray_pneumonia_model.h5")
print("\nModel saved as xray_pneumonia_model.h5")

# --------------------------------------------
# 8. Plot Accuracy & Loss
# --------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.show()