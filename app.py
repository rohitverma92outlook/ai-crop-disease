import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from io import BytesIO

option = st.sidebar.radio('Crop',["Pepper","Potato","Rice","Tomato"])


if option =="Pepper":
    st.title("Prediction of Pepper Crop Disease")


    uploaded_image = st.file_uploader("Please upload a Pepper crop image",type=['png','jpeg','jpg'])
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Pepper crop photo')

    model = tf.keras.models.load_model("saved_models/pepper_version_3")
    class_names = ['Pepper Bell Bacterial Spot', 'Pepper Bell Healthy']

    
    image = np.array(image)
    image = tf.image.resize(image,[128,128])
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.subheader("Artificial Intelligence Model Prediction")
    if st.button("Submit"):
        st.write(f"Predicted Category is {predicted_class} with {confidence*100}% accuracy")
    else:
        st.write("Click 'Submit'")

if option == "Potato":
    st.title("Prediction of Potato Crop Disease")

    data = st.file_uploader("Please upload a Potato crop image",type=['png','jpeg','jpg'])
    uploaded_image = Image.open(data)
    st.image(uploaded_image, caption='Uploaded Potato crop image')

    model = tf.keras.models.load_model("saved_models/potato_version_1")
    class_names = ['Early Blight', 'Late Blight', 'Healthy']

    image = np.array(uploaded_image)
    image = tf.image.resize(image,[128,128])
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.subheader("Artificial Intelligence Model Prediction")
    if st.button("Submit"):
        st.write(f"Predicted Category is {predicted_class} with {confidence*100}% accuracy")
    else:
        st.write("Click 'Submit'")


if option =="Rice":
    st.title("Prediction of Rice Crop Disease")

    data = st.file_uploader("Please upload a Rice crop image",type=['png','jpeg','jpg'])
    uploaded_image = Image.open(data)
    st.image(uploaded_image, caption='Uploaded Rice crop photo')

    model = tf.keras.models.load_model("saved_models/rice_version_1")
    class_names = ['Rice Brown Spot', 'Rice Healthy', 'Rice Hispa', 'Rice Leaf Blast']

    image = np.array(uploaded_image)
    image = tf.image.resize(image,[128,128])
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.subheader("Artificial Intelligence Model Prediction")
    if st.button("Submit"):
        st.write(f"Predicted Category is {predicted_class} with {confidence*100}% accuracy")
    else:
        st.write("Click 'Submit'")


if option =="Tomato":
    st.title("Prediction of Tomato Crop Disease")

    data = st.file_uploader("Please upload a Tomato crop image",type=['png','jpeg','jpg'])
    uploaded_image = Image.open(data)
    st.image(uploaded_image, caption='Uploaded Tomato crop photo')

    model = tf.keras.models.load_model("saved_models/tomato_version_1")
    class_names = ['Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites Two Spotted Spider Mite',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Tomato Healthy']

    image = np.array(uploaded_image)
    image = tf.image.resize(image,[128,128])
    img_batch = np.expand_dims(image, 0)
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.subheader("Artificial Intelligence Model Prediction")
    if st.button("Submit"):
        st.write(f"Predicted Category is {predicted_class} with {confidence*100}% accuracy")
    else:
        st.write("Click 'Submit'")