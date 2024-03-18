import os
from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import cv2

capture = cv2.VideoCapture(0)
detectHand = HandDetector(maxHands=1)
while True:
    success, img = capture.read() #reading through camera
    hands, img =detectHand.findHands(img)
    cv2.imshow("Tasveer",img) #showing whatever our camera read
    cv2.waitKey(1)