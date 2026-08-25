"""
train_model.py
----------------
Trains a CNN on the CIFAR-10 dataset (10 classes of small 32x32 images)
and saves the trained model as cifar10_model.h5 so it can be loaded by
the Streamlit app (app.py) for live predictions on uploaded images.

Run this once locally (or in Colab with a GPU) before deploying the
Streamlit app:

    python train_model.py

It will create: cifar10_model.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1. Load & prepare the dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -----------------------------
# 2. Build the CNN model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# 3. Data augmentation (helps a lot on CIFAR-10)
# -----------------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

# -----------------------------
# 4. Train
# -----------------------------
EPOCHS = 30
BATCH_SIZE = 64

history = model.fit(
    datagen.flow(x_train, y_train_cat, batch_size=BATCH_SIZE),
    validation_data=(x_test, y_test_cat),
    epochs=EPOCHS,
    steps_per_epoch=len(x_train) // BATCH_SIZE,
)

# -----------------------------
# 5. Evaluate & save
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

model.save("cifar10_model.h5")
print("Saved model to cifar10_model.h5")
