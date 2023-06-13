from function import *

import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()
from PIL import Image

###### Design App #########

import time

st.title('Dog Breed Prediction')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static/images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0

uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # Perform image feature extraction (with loading chart)
        with st.spinner('Calculating...'):
            time.sleep(2)  # Simulating feature extraction process
            prediction = predictor(os.path.join('static/images', uploaded_file.name))

        os.remove('static/images/' + uploaded_file.name)
        # deleting uploaded saved picture after prediction
        st.write("Predictions:")
        st.write(prediction)
