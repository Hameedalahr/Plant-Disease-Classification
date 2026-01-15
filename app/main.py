import os
import json
from PIL import Image

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download

working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = f"{working_dir}/trained_model/trained_model.h5"

model_path = hf_hub_download(
    repo_id = "Hameedalahr/plants_trained_model",
    filename = "trained_model.h5"
)
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))


def load_and_process_image(image_path, target_size=(224,224)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array,axis=0)
    img_array = img_array/255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_process_image(image_path)
    predictions = model.predict(preprocessed_img)
    prediction_class_index = np.argmax(predictions,axis = 1)[0]
    predicted_class_name = class_indices[str(prediction_class_index)]
    return predicted_class_name

st.title("🍀Plant Disease Classification")

uploaded_image = st.file_uploader("Upload an Image...", type=["png","jpg","jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1 ,col2 = st.columns(2)

    with col1:
        resized_img  = image.resize((150,150))
        st.image(resized_img)
    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image,class_indices)
            st.success(f"Prediction : {str(prediction)}")