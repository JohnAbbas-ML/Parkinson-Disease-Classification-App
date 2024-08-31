import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import zipfile
import random

model = tf.keras.models.load_model('parkinson_detection_model.h5')

class_names = ['Mild-Demented', 'Moderate-Demented', 'Non-Demented', 'Severe-Demented', 'Very-Mild-Demented']

def preprocess_image(image):
    image = image.resize((150,150))
    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0) #image.shape --> 1*150*150*3

    return image

st.title("Parkinson Disease Classification App - A project by Team Muhammad John Abbas")
st.write("Upload an MRI image of the brain to classify")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption='Uploaded image', use_column_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)

    confidence = np.max(prediction) * 100 #95.362569755 --> .2f 95.36

    st.write(f"Prediction:**{class_names[predicted_class[0]]}**")

    st.write(f"Confidence:**{confidence:.2f}**")


def create_sample_zip():
    sample_images_dir = "sample_images"

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for root , i , files in os.walk(sample_images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, os.path.basename(file_path))
    return zip_buffer.getvalue()

if st.button("Download Testing Images"):
    zip_data = create_sample_zip()
    st.download_button(label="Download Zip of Testing Images",
                       data = zip_data,
                       file_name="testing_images.zip",
                       mime="application/zip")

def display_result_images():
    results_dir = 'Results'

    image_files = [f for f in os.listdir(results_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    selected_images = random.sample(image_files, 6)

    st.write("### Model Results")

    cols = st.columns(3)

    for i, image_files in enumerate(selected_images):
        image_path = os.path.join(results_dir, image_files)
        image = Image.open(image_path)

        cols[i % 3].image(image, caption=image_files, use_column_width=True)
if st.button("Display Results"):
    display_result_images()
