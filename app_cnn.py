import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

cnn_model = load_model('./facial_expresion_cnn_model.h5')

st.title('Expression Detection using CNN')

def image_preprocessing(input_image, img_tgt_size=(128, 128)):
    image = Image.open(input_image).convert('RGB')
    image = image.resize(img_tgt_size)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

image_input_file = st.file_uploader("Upload an image to detect", type=["jpg", "jpeg", "png"])
if image_input_file is not None:
    st.image(image_input_file, caption='Uploaded Image', use_column_width=True)
    preprocessed_image = image_preprocessing(image_input_file)
    prediction = cnn_model.predict(preprocessed_image)
    class_names = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral']
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f'Predicted expression: {predicted_class}')
