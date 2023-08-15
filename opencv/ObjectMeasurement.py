import cv2
import numpy as np
import Utilities as ut
import sys, signal
from matplotlib import pyplot as plt
import ctypes

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

testing = True
webcam = False
path = '1.jpg'
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

while testing:
    if webcam: 
        success, img=cap.read()
        cv2.imshow('Cam feed', img)
        cv2.waitKey(1)
    else: 
        #originalImg = cv2.imread(path)
        #dim = ut.resizeImage(originalImg)
        #originalImg = cv2.resize(originalImg, dim, interpolation = cv2.INTER_AREA)

        originalImg = ut.getContours(path)

        originalImg = cv2.resize(originalImg, dsize=(ctypes.windll.user32.GetSystemMetrics(0) // 2, ctypes.windll.user32.GetSystemMetrics(1) // 2))
        cv2.imshow('Original', originalImg)
        cv2.waitKey(1)

# TODO: Een manier vinden om afstand te berekenen om de juiste afmetingen te vinden.
# NOTE: Start met toepassen van de grootte van het A4tje 
# NOTE: Idealiter een check op aanwezige objecten om vanuit bestaande gegevens (die worden opgezocht)
# NOTE: de rest van het geheel te schalen 
