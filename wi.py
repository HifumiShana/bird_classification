import streamlit as st
import numpy as np
from load_model import model
import os
import matplotlib.image as mpimg
from tensorflow.keras.applications.resnet50 import preprocess_input 

model.load_weights('D:\\AI\\Project\\weights_resnet_acc.hdf5')

st.title('BIRDS CLASSIFICATION APP')
st.write('260 Species of Birds:')
birds_species = os.listdir('D:\\AI\\Project\\archived\\consolidated')
birds_species = sorted(birds_species)
diction = dict()
for i in range(10):
    diction['collumn '+str(i)] = list()
for i in range(10):
    for j in range(26):
        diction['collumn '+str(i)].append(birds_species[(26*i+j)])
st.write(pd.DataFrame(diction))

opt = st.selectbox(options=birds_species, label='Choose a species to see')
img = mpimg.imread(f'D:\\AI\\Project\\archived\\test\\{opt}\\1.jpg') 
st.image(img)

uploaded_img = st.file_uploader(label="Choose a image of a bird you want to know the names")
if uploaded_img != None:
    img = mpimg.imread(uploaded_img)
    img = preprocess_input(img)
    img = img.reshape(1,224,224,3)
    output = model.predict(img)
    output = birds_species[output.argmax()]
st.write('*The image shows the: *') 
if uploaded_img != None:
    st.write(output)