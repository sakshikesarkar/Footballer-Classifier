import streamlit as st
from utils import set_background
from PIL import Image
import torch
from classifier import classifier
from io import BytesIO
import base64
from Model import CNN

st.set_page_config(
    page_title='Footballer Classification',
    layout='centered'
)

set_background('utils/bg.jpg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 60px;
        color: #f0f0f0;
        font-weight: bold;        
        margin-top: -75px;
    }
    .header {
        display: flex;
        justify-content: center;  /* Center horizontally */
        align-items: center;  /* Center vertically (if needed) */
        text-align: center;  
        font-size: 30px;
        color: #87cefa;
        white-space: nowrap;
        margin-top: -20px;
        width: 100%;  /* Ensures full width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Footballer Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as Cristiano Ronaldo, Lionel Messi and Neymar.</div>', unsafe_allow_html=True)

file = st.file_uploader('',type = ['jpg','jpeg','png','jfif'])

model = torch.load('Model/best_model.pth',map_location=torch.device('cpu'))
model.eval()  # Set to evaluation mode

class_names = {0:'Cristiano Ronaldo', 1:'Lionel Messi', 2:'Neymar'}

if file is not None:
    
    image = Image.open(file).convert('RGB')

    prediction, score = classifier(image, model, class_names)
    
    if score == 'Error':
        # st.error(prediction)
        st.markdown(
                f"""
                <div style="color: black; font-size: 18px; font-weight: bold; background-color: #f8d7da; padding: 10px; border-radius: 5px; text-align: center; text-align: left;">
                    <p style="color: red;"> {prediction} </p>
                    <p><strong>Image should be:</strong></p>
                    <p style="color:green">1. Having only one face</p>
                    <p style="color:green">2. Face should be clear</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.stop()
    
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()

    # Display classification results with reduced gap and no extra space
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">        
        <img src="data:image/png;base64,{img_base64}" style="width: 400px; height: 370px; object-fit: cover; margin-top: -20px;"/>
        <div style="font-size:40px; font-weight:bold; margin-left: 20px; color:#FFFFFF; white-space: nowrap;">
            <p> <strong>Result: {prediction}</strong></p>
            <p style="margin-top:-10px;"> <strong> Score: {score}% </strong> </p>
        </div>        
        </div>
        """,
        unsafe_allow_html=True
    )
