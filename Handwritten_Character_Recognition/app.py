import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


model = tf.keras.models.load_model("models/digit_model.h5")

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a **single digit (0-9)** written on white paper. Letters are not supported.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image.thumbnail((20, 20), Image.LANCZOS)
    new_image = Image.new('L', (28, 28), color=0)
    left = (28 - image.width) // 2
    top = (28 - image.height) // 2
    new_image.paste(image, (left, top))
    img_array = np.array(new_image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)
    img_array = preprocess_image(image)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_digit}")
