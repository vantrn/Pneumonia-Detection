import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model saved in the new Keras format
model = load_model('model/pneumonia_model.keras')

# Path to test folder (choose either NORMAL or PNEUMONIA to test)
test_dir = 'data/test/PNEUMONIA'  # or 'data/test/NORMAL'
img_size = (150, 150)

# List test images
images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Predict on a few test images
for img_name in images[:20]:  # Adjust number as needed
    img_path = os.path.join(test_dir, img_name)
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get prediction
    prediction = model.predict(img_array)[0][0]
    label = 'PNEUMONIA' if prediction > 0.5 else 'NORMAL'
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Display result
    plt.imshow(img)
    plt.title(f"Prediction: {label} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()