from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/model.keras")

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))  # ← use the size you trained your model with!
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # (1, 64, 64, 3)
    return image_array

def predict(image_array):
    prediction = model.predict(image_array)
    confidence = prediction[0][0]
    label = 1 if confidence >= 0.5 else 0
    return label, confidence
