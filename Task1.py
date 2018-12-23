
import numpy as np
import cv2
import argparse
import csv
import os

# Argument Parser

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Load cascade classifier for face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform classification
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

     #print('Number of faces:', len(faces))
print('Filename: ', args["image"],' Number of faces:', len(faces))

cv2.imshow('img',img)
cv2.waitkey(0)


#for imagePath in os.listdir('dataset'):

     # Writing to csv file 

     # with open('Task1.csv', mode='w') as Task1:
     #    TaskWriter = csv.writer(Task1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # TaskWriter.writerow([args["image"]] + [str(len(faces))])





