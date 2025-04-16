import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load the trained model
model = load_model("D:/test_model/model/cat_dog_model.keras")

# Function to predict image class
def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = img_array / 255.0  # Normalize

    # Prediction
    prediction = model.predict(img_array)
    label = 'Dog 🐶' if prediction[0][0] >= 0.5 else 'Cat 🐱'
    confidence = round(prediction[0][0], 3) if label == 'Dog 🐶' else round(1 - prediction[0][0], 3)
    
    print(f"\nPrediction: {label} (Confidence: {confidence * 100}%)")

# Example usage
predict_image("sample.jpg")
