import os
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import cv2
import numpy as np
import time

capture = cv2.VideoCapture(0)
detectHand = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data/B"
cnt = 0

while True:
    success, img = capture.read() #reading through camera
    hands, img =detectHand.findHands(img) #here the hands variable is array of total number of hands within which each hand is a dictionary data structure which stores various information about each hand
    if hands:
        hand=hands[0]#as there can be multiple hands in the hand array so we are taking only 1 hand i.e hands[0]
        x,y,w,h=hand['bbox']#as we know hand is dictonary which contains the description of hand so within that there's a tuple having 4 elements length,breadth,height,width which are fixed that's why tuple is used rather list and the name of the tuple is bbox
        imgCrop=img[y-offset:y+h+offset, x-offset:x+w+offset] #creating a new image which is rectangle is liye to sirf 2 parameters jaa raha hai and we know that rectangle mein 2 hi elements rehta hai ek length and breadth
        cv2.imshow('Cropped_Image',imgCrop) #showing the cropped image with dimensions
        #Now we will create our own image of fixed dimension with white background and we will insert our hand there

        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 #so we know image are of matrix type so we'll be creating a matrix which can be done by numpy and keep it of fixed length i.e 300 and then intialize all the values with 255 so it becomes white but we can only initialize an array with either 0 or 1 so we'll initialize it all with 1 and the multiply the matrix by 255 and to initialize a matrix numpy has an inbuilt function as np.zeros() or np.ones respectively
        #np.ones()->initialize array with all 1's
        #1st param i.e (imgSize,imgSize) -> row and columns shape/size details
        #2nd param i.e np.uint8 -> initialize datatype as UnsignedInteger_of_8bits because initially it will be of float datatype
        #*255 is used to make the image white from black as we know all 1's will be black only

        #Now overlay our hand image to imgWhite
        imgCropShape=imgCrop.shape #any image's shape is an array of 3 elements height,width,channels(idk what channels are that's none of my concern )
        height_of_cropped_img = imgCropShape[0]
        width_of_cropped_img = imgCropShape[1]

        imgResize=cv2.resize(imgCrop,(300,300))
        imgWhite[0:imgResize.shape[0],0:imgResize.shape[1]]=imgResize
        #0:height_of_cropped_img,0:width_of_cropped_img by wrting this we are specifying from where to where our image should spread so it is from 0th(0th row) position i.e top position in a matrix to the height of the image and from 0th column i.e from leftmost position of a matrix to the width of the image

        #Centering the imgWhite

        cv2.imshow('Created_Image', imgWhite)
    cv2.imshow("Sign Language Prediction of Kids Using  Machine Learning ",img) #showing whatever our camera read
    key = cv2.waitKey(1)
    if key == ord("s"):
        cnt += 1
        cv2.imwrite(f'{folder}/Image_{cnt}.jpg',imgWhite) #here we wrote folder in which image will be saved after \ we wrote name of image with current time as it will be unique for every image
        print(cnt)


