# Task1.py - Reads in image and detects faces. If faces are present, they are saved into a new directory.

import numpy as np
import cv2
import argparse
import csv
import os
import glob

# Argument Parser

#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

# Load cascade classifier for face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# initialise data set for upload to csv file
csvData = []

#  Perform classification
def faceClass(image):
     faces = face_cascade.detectMultiScale(image, 1.05, 3)
     #for (x,y,w,h) in faces:
          #cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
     #cv2.imshow('image',image)
     #cv2.waitKey(0)
     print('Faces found: ',len(faces))
     return len(faces)

# Read in image
#def readImage(image):
     #greyImg = cv2.imread(image, 0)
     #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     #return greyImg

img_dir = "dataset" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)

# Make directory for face and noisy images
os.mkdir('FaceImages')
os.mkdir('NoisyImages')

# Loop through dataset performing classification
for f1 in files:
     filename = os.path.basename(f1)
     img = cv2.imread(f1)
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     #img = readImage(imagePath)
     numFace = faceClass(gray)
     if numFace > 0:
          numVal = 1
          cv2.imwrite(os.path.join('FaceImages', filename), img)
     else:
          numVal = -1
          cv2.imwrite(os.path.join('NoisyImages', filename), img)
     csvData.append([filename,numVal])
     print('Filename: ', filename,' Number of faces:', numFace)

csvData.sort()
     
# Writing to csv file 
with open('Task1.csv', mode='w') as Task1:
     TaskWriter = csv.writer(Task1, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
     TaskWriter.writerows(csvData)

Task1.close()





