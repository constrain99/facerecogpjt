import cv2
import numpy as np
import face_recognition


imgelon=face_recognition.load_image_file('imageBasics/pran_img.jpg')
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB)
imgtest=face_recognition.load_image_file('imageBasics/elon_test.jpg')
imgtest=cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


faceLoc=face_recognition.face_locations(imgelon)[0]
encodeelon=face_recognition.face_encodings(imgelon)[0]
cv2.rectangle(imgelon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoctest=face_recognition.face_locations(imgtest)[0]
encodetest=face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


result=face_recognition.compare_faces([encodeelon],encodetest)
faceDis=face_recognition.face_distance([encodeelon],encodetest)
print(result,faceDis)
cv2.putText(imgtest,f'{result}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow('elon musk',imgelon)
cv2.imshow('elon musk',imgtest)
cv2.waitKey(0)