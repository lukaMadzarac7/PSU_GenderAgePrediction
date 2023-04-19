import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split

def mae(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

path = "UTKFace"
images = []
age = []
gender = []

print("Ulazak u for")
for img in os.listdir(path):
  ages = img.split("_")[0]
  genders = img.split("_")[1]
  img = cv2.imread(str(path)+"/"+str(img))
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  images.append(np.array(img))
  age.append(np.array(ages))
  gender.append(np.array(genders))

print("Np convert")
  
age = np.array(age,dtype=np.int64)
images = np.array(images) 
gender = np.array(gender,np.uint64)

print("Splitanje")

x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(images, age, random_state=42)

x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(images, gender, random_state=42)
print("Gotov sam sa splitanjem")



##################################################
#Model za dob
##################################

print("Ulazim u definiranje gender modela")

age_model = Sequential()
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))             
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))
age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
age_model.add(MaxPool2D(pool_size=3, strides=2))

age_model.add(Flatten())
age_model.add(Dropout(0.2))
age_model.add(Dense(512, activation='relu'))


age_model.add(Dense(1, activation='linear', name='age'))             
age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print(age_model.summary())         
                           
fit_age = age_model.fit(x_train_age, y_train_age, validation_data=(x_test_age, y_test_age), epochs=10, batch_size=16)

age_model.save('dob_model4.h5')

print('Gotov sam s modelom i spremit cu ga')
print('Zadnja linija')



################################################################
#Model za spol
##################################################
gender_model = Sequential()

gender_model.add(Conv2D(36, kernel_size=3, activation='relu', input_shape=(200,200,3)))

gender_model.add(MaxPool2D(pool_size=3, strides=2))
gender_model.add(Conv2D(64, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(128, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(256, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Conv2D(512, kernel_size=3, activation='relu'))
gender_model.add(MaxPool2D(pool_size=3, strides=2))

gender_model.add(Flatten())
gender_model.add(Dropout(0.2))
gender_model.add(Dense(512, activation='relu'))
gender_model.add(Dense(1, activation='sigmoid', name='gender'))

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fit_gender = gender_model.fit(x_train_gender, y_train_gender, validation_data=(x_test_gender, y_test_gender), epochs=10, batch_size = 16)

gender_model.save('spol_model3.h5')

print('Gotov sam s modelom i spremit cu ga')
print('Zadnja linija')



############################################################
#Testiranje preciznosti modela za spol
####################################################################

from keras.models import load_model
my_model = load_model('spol_model.h5', compile=False)


predictions = my_model.predict(x_test_gender)
y_pred = (predictions>= 0.5).astype(int)[:,0]

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test_gender, y_pred))

#from sklearn.metrics import confusion_matrix
#import seaborn as sns
#cm=confusion_matrix(y_test_gender, y_pred)  
#sns.heatmap(cm, annot=True)

############################################################
#Testiranje modela za dob
####################################################################
my_model = load_model('dob_model.h5', compile=False)

predictions = my_model.predict(x_test_age)
y_pred = (predictions>= 0.5).astype(int)[:,0]

print ("Mae = ", mae(y_test_age, y_pred))