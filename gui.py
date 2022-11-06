import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
import cv2
import PIL.Image, PIL.ImageTk
from threading import Thread
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib


face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
pca = joblib.load('models/pca1.joblib')
classifier = joblib.load('models/svm1.joblib')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

root = tk.Tk()
root.title('Emotion Detection')

video = cv2.VideoCapture(0)

canvas_w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
canvas_h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

canvas = Canvas(root, width=canvas_w, height=canvas_h, bg='white')
canvas.pack()

label = Label(root, text='None', font=('courier', 18))
label.pack()

frame = None
photo = None
count = 0

def run_classification():
    global label, frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            imVec = roi_gray.reshape(1, 48*48)
            #roi = np.expand_dims(roi,axis=0)
            img_pca = pca.transform(imVec)
            prediction = classifier.predict(img_pca)
            result = int(prediction[-1])
            lbl = emotion_labels[result]
            print(lbl)
            label.configure(text=lbl)
        else:
            label.configure(text='No face') 

def update_frame():
    global canvas, photo, count, frame
    ret, frame = video.read()
    #frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    thread = Thread(target=run_classification)
    thread.start()
    root.after(15, update_frame)

update_frame()

root.mainloop()