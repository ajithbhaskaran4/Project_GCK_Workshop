import streamlit as st
from BackEnd.Image2PointCloud import CNN_Prediction, Image2PointCloud
from PIL import Image
import numpy as np
import tempfile
import base64
import os
import io


st.set_page_config(layout="wide")

@st.cache_resource
def getBrainImage():
    return Image.open(r"BackEnd/college_Image.jpg")
    
    
def prevButton():
    if not (st.session_state.sliderPos == 1):
        st.session_state.sliderPos = st.session_state.sliderPos -1
        
def nextButton():
    if not (st.session_state.sliderPos >= st.session_state.NumImages):
        st.session_state.sliderPos = st.session_state.sliderPos +1

if 'flag' not in st.session_state:
    st.session_state.flag = False
    
if 'CNN' not in st.session_state:
    st.session_state.CNN = CNN_Prediction()

if 'sliderPos' not in st.session_state:
    st.session_state.sliderPos = 1

if 'backend' not in st.session_state:
    st.session_state.backend = Image2PointCloud()

st.title("MRI CYST ANALYSIS")

st.image(getBrainImage())

st.text("An Application to view and quantify the presence of cyst from MRI")


def change_MRI():
    #st.subheader("MRI Image")
    currentImage = st.session_state.ImageStack[st.session_state.MRI_Slider, :,:,:]
    #print("current Image size: ", currentImage.shape)
    currentImage = Image.fromarray(np.uint8(currentImage), mode = "RGB")
    st.image(currentImage)
            
uploaded_files  = st.file_uploader("Choose the MRI Images", accept_multiple_files=True, key = 'Folder_Path')

if uploaded_files:
    temp_dir = tempfile.mkdtemp()
    st.write("Uploaded Files:")
    for file in uploaded_files:
        file_name = file.name
        file_path = os.path.join(temp_dir, file_name)
        # Save the uploaded file with its original name to the temporary directory
        with open(file_path, "wb") as f:
            f.write(file.read())
    st.session_state.backend.setPaths(temp_dir)


if st.button('Submit'):
    
    st.session_state.backend.read_mri_images()
    st.session_state.NumImages = st.session_state.backend.getnumberofImages()
    #if 'ImageStack' not in st.session_state:
    st.session_state.ImageStack = st.session_state.backend.get_StackMRI()
    #if 'prediction' not in st.session_state:
    st.session_state.prediction = st.session_state.CNN.predictCNN(st.session_state.ImageStack)
       
    print("received all data")
    st.session_state.flag = True

if st.session_state.flag == True:
    ReportString = str(st.session_state.NumImages) + " MRI Images are Uploaded and processed"
    st.write(ReportString)
    prev, label, Next = st.columns([1,2,1])
    with prev:
        st.button("Previous", on_click = prevButton)
    with Next:
        st.button("Next", on_click = nextButton)
    with label : 
        string = "       MRI Image Number :" + str(st.session_state.sliderPos)
        st.write(string)
    #st.session_state.sliderPos = st.slider("Select MRI Image Slice", min_value=1, max_value=st.session_state.NumImages, step=1, key = "MRI_Slider")
    mri_image, MRI_Cyst = st.columns([1,1])
    with mri_image:
        st.write("Original MRI Image")
        currentImage = Image.fromarray(np.uint8(st.session_state.backend.getMRIImage(st.session_state.sliderPos-1)), mode = "RGB")
        st.image(currentImage)
    with MRI_Cyst:
        st.write("Processed MRI Image. ** red region Indicates the presence of cyst")
        mriImage = np.uint8(st.session_state.backend.getMRIImage(st.session_state.sliderPos-1))
        cystPos = np.where(st.session_state.prediction[st.session_state.sliderPos-1,:,:,0]==255)
        RLayer = mriImage[:,:,0]
        RLayer[cystPos] = 255
        GLayer = mriImage[:,:,1]
        GLayer[cystPos] = 0
        BLayer = mriImage[:,:,2]
        BLayer[cystPos] = 0
        currentPred = Image.fromarray(mriImage, mode = 'RGB')
        st.image(currentPred)
   
            