import cv2 
import numpy as np   

cascade=cv2.CascadeClassifier(r'C:\Users\Administrator\Documents\For Python\Face recognition Test\Face_Recognition_Test\haarcascade_frontalface_default.xml')
image=cv2.VideoCapture(0)

ret, img = image.read()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, 1.4, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(256,0,0),2)

cv2.imshow('Face detection 1#:',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

