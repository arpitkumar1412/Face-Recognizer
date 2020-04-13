# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:45:15 2020

@author: Arpit
"""
import cv2
import os
import numpy as np
import face_reg as fr

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Arpit\Desktop\face_reg\trainingData.yml')

cap = cv2.VideoCapture(0)

name = {0: "priyanka", 1: "kangana"}

while True:
    ret,test_img = cap.read()
    faces_detected, gray_img = fr.faceDetection(test_img)
    
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img, (x,y),(x+w,y+h),(255,0,0),thickness = 1)

    resized_img = cv2.resize(test_img,(500,300))
    cv2.imshow('face detection', resized_img)
    cv2.waitKey(0)
    
    for face in faces_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(roi_gray)
        
        print("confidence:", confidence)
        print("label:", label)
        fr.draw_rect(test_img, face)
        predicted_name = name[label]
        if confidence < 60:
            fr.put_text(test_img, predicted_name,x,y)
    
    resized_img = cv2.resize(test_img, (1000,700))
    cv2.imshow("face", resized_img)
    if(cv2.waitKey(10) == ord('q')):
        break;
        
cap.release()
cv2.destroyAllWindows






