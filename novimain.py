from cv2 import CAP_ARAVIS
import math
from keras.models import load_model
from time import sleep
from skimage import io
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from keras.preprocessing import image
import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.write('-------Predikcija dobi i spola-------')

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_model = load_model('dob_model.h5')
gender_model = load_model('spol_model.h5')

gender_labels = ['Muško', 'Žensko']

st.write("## Učitajte sliku koja sadrži lice:")

cap = st.file_uploader("")


if cap is not None:

    image = Image.open(cap)
    faces = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY))
    imaz=cv2.imread('temp.jpg')
    gray=cv2.cvtColor(faces,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(imaz,1.3,5)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(imaz,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        roi=roi_gray.astype('float')/255.0  
        roi=np.array(roi)
        roi=np.expand_dims(roi,axis=0) 
    
    im_color=imaz[y:y+h,x:x+w]

    if(type(im_color) == type(None)):
        pass
    else:
        im_color=cv2.resize(im_color,(200,200), cv2.INTER_AREA)

    

    #Dob
    gender_predict = gender_model.predict(np.array(im_color).reshape(-1,200,200,3))
    gender_predict = (gender_predict>= 0.5).astype(int)[:,0] 
    gender_label=gender_labels[gender_predict[0]] 
    st.write(f'Spol: {gender_label}')
        
    #Spol
    age_predict = age_model.predict(np.array(im_color).reshape(-1,200,200,3))
    age = round(age_predict[0,0])
    st.write(f'Dob: {age}')

    st.image(cap, caption='Unešena slika')
        
   