# train_model.py
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the CSV
train_df = pd.read_csv('/Users/nandagopal/Downloads/plant-pathology-2020-fgvc7/train.csv')

# Check what's inside
print(train_df.head())

# Prepare the data
train_images = []
train_labels = []

for index, row in train_df.iterrows():
    img_path = os.path.join('images', row['image_id'] + ".jpg")
    img = load_img(img_path, target_size=(64, 64))
    img = img_to_array(img)
    img = img / 255.0  # normalize
    train_images.append(img)

    if row['healthy'] == 1:
        label = 0  # healthy
    else:
        label = 1  # diseased

    train_labels.append(label)

X = np.array(train_images)
y = np.array(train_labels)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('plant_disease_cnn_model.h5')
print("✅ Model saved as plant_disease_cnn_model.h5")
