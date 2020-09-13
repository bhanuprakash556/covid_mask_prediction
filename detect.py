# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 16:56:49 2020

@author: bhanu prakash
"""
import tensorflow.keras
#from PIL import Image, ImageOps
import cv2
import numpy as np

# Disable scientific notation for clarity
#np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source=cv2.VideoCapture(0)
#source.set(3,224)
#source.set(4,224)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

###########################################################

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  
    

    for (x,y,w,h) in faces:
            
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            face_img=img[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(224,224))
            normalized_image_array = (resized.astype(np.float32) / 127.0) - 1
            #normalized=resized/255.0
            #reshaped=np.reshape(normalized,(1,224,224,3))
            data[0] = normalized_image_array
         
            result=model.predict(data)
             
            label=np.argmax(result,axis=1)[0]
            print(label)          
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
            
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()


