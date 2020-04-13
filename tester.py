# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:34:33 2020

@author: Arpit
"""

import cv2
import os
import numpy as np
import face_reg as fr

test_img = cv2.imread(r'C:\Users\Arpit\Desktop\face_reg\test_images\download (2).jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)

faces,faceID = fr.labels_for_training_data(r'C:\Users\Arpit\Desktop\face_reg\train_images') 
print(faceID)
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.save('trainingData.yml')

#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read(r'C:\Users\Arpit\Desktop\face_reg\trainingData.yml')

name = {0: "priyanka", 1: "kangana", 2: "arpit"}

for face in faces_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if confidence < 40:
        fr.put_text(test_img, predicted_name,x,y)
    
resized_img = cv2.resize(test_img, (1000,700))
cv2.imshow("face", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows






