import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import streamlit as st

model = load_model('models/digit_model.h5')
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  
    img = img.resize((28, 28))  
    img_array = np.array(img)  
    img_array = img_array / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1)  
    return img_array

def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)  
    return predicted_digit

st.title("Handwritten Digit Recognition")

st.write("Upload an image of a handwritten digit (0-9):")
image_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if image_file is not None:
    with open("uploaded_image.png", "wb") as f:
        f.write(image_file.getbuffer())
    predicted_digit = predict_digit("uploaded_image.png")
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Digit: {predicted_digit}")

