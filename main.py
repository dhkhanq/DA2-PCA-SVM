from time import sleep
import cv2
import numpy as np
from time import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib


face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
pca = joblib.load('models/pca1.joblib')
classifier = joblib.load('models/svm1.joblib')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.rectangle(frame,(x-2,y),(x+110,y-25),(255,0,0), cv2.FILLED)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            imVec = roi_gray.reshape(1, 48*48)
            img_pca = pca.transform(imVec)
            prediction = classifier.predict(img_pca)
            result = int(prediction[-1])
            label = emotion_labels[result]
            print(label)
            label_position = (x,y-7)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()