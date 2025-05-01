import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Paths
data_dir = 'data'
train_path = os.path.join(data_dir, 'train')
val_path = os.path.join(data_dir, 'val')
test_path = os.path.join(data_dir, 'test')

# Image preprocessing
img_size = (150, 150)
batch_size = 32

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path, target_size=img_size, batch_size=batch_size, class_mode='binary')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path, target_size=img_size, batch_size=batch_size, class_mode='binary')

# Build model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# ✅ Save model in new format after it's fully built
model.save('model/pneumonia_model.keras')