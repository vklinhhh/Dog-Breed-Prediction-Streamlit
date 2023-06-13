from function import *

#importing all the helper fxn from helper.py which we will create later

import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()
from PIL import Image
st.title('Dog Breed Classifier')

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

        prediction = predictor(os.path.join('static/images',uploaded_file.name))

        os.remove('static/images/'+uploaded_file.name)

        # deleting uploaded saved picture after prediction
        st.write("Predictions:")
        st.write(prediction)

