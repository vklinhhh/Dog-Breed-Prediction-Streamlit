from function import *

import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()
from PIL import Image
import streamlit.components.v1 as components
###### Design App #########

import time
import base64
from load_css import local_css
def set_bg_hack_url():
    '''
    A function to unpack an image from a URL and set it as the background.
    Returns:
    The background.
    '''
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://images.unsplash.com/photo-1597633611385-17238892d086?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_hack_url()

local_css("style.css")

st.markdown(
    """
    <h1 style='text-align: center; color: #1d3557;'>Dog Breed Predictor</h1>
    """,
    unsafe_allow_html=True
)

### predictor

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static/images', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

x = "<div style='text-align: center;'><span class='customcolor'><span class='bold'> \
        Hi, welcome to my App! <br> Please upload your pet (dog, yes) and wait a minute so my machine can predict your buddy. <br> \
        This app is belong to VKLinh. </span></span></div>"


st.markdown(x, unsafe_allow_html=True)
uploaded_file = st.file_uploader('Uploaded image', label_visibility='collapsed')

# text over upload button "Upload Image"
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the image
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        if st.button("Predict this image"):
            # Perform image feature extraction (with loading chart)
            with st.spinner(':blue[Wait a moment nha!...]'):
                time.sleep(2)  # Simulating feature extraction process
                prediction = predictor(os.path.join('static/images', uploaded_file.name))

            os.remove('static/images/' + uploaded_file.name)
            # deleting uploaded saved picture after prediction
            # Display the prediction within the HTML layout
            st.markdown(
            f"""
            <div style='text-align: center; color: #1d3557'>
                <h2 style='color: #1d3557;'>Woo hoo! You got a</h2>
                <h2 style='color: #1d3557;'><span class='highlight red'><strong>{prediction}</strong></span></h2>
            </div>
            """,
        unsafe_allow_html=True
    )

