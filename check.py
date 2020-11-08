import cv2
import numpy as np
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

model = tf.keras.models.load_model(input("model file name:"))

cam = cv2.VideoCapture(0)

people = []#list of people's names
face = ""
count = 0
while(True):
    ret, frame = cam.read()

    if ret == True:
        faces = face_cascade.detectMultiScale(frame, 1.3, 3)
        frame1 = frame

        for (x,y,w,h) in faces:
            test = cv2.resize(frame[y:y+h, x:x+w], (100, 100), interpolation = cv2.INTER_AREA)
            frame1 = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            person = model.predict_classes(np.array([test]))
            frame1 = cv2.putText(frame1, people[person[0]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
            print(people[person[0]])

        cv2.imshow('frame', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cam.release()