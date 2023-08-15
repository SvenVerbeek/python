import cv2
import numpy as np
import ctypes
from pathlib import Path
from typing import List, Union, Callable

img_path = '1.jpg'

def rescaleImage(img):
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return dim

def nothing(x):
    pass

def adjustingEnabled(s):
    return False if s == 0 else True

def show_img(img_list: Union[np.ndarray, List[np.ndarray]], combine_fun: Callable = np.vstack,
             window_name='demo', window_size=(ctypes.windll.user32.GetSystemMetrics(0) // 2, ctypes.windll.user32.GetSystemMetrics(1) // 2),
             delay_time=0, note: Union[str, List[str]] = None, **options):
    if isinstance(img_list, np.ndarray):
        img_list = [img_list]

    if isinstance(note, str):
        print(note)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if window_size:
        w, h = window_size
        cv2.resizeWindow(window_name, w, h)

    result_list = []
    for idx, img in enumerate(img_list):
        img = np.copy(img)
        if note and isinstance(note, list) and idx < len(note):
            cv2.putText(img, note[idx], org=options.get('org', (50, 50)),
                        fontFace=options.get('fontFace', cv2.FONT_HERSHEY_SIMPLEX),
                        fontScale=options.get('fontScale', 2), color=(0, 255, 255), thickness=4)
        result_list.append(img)
    cv2.imshow(window_name, combine_fun(result_list))
    cv2.waitKey(delay_time)

def getContours(path): #, 
                #Thr=[50,200], 
                #showCanny=False, 
                #minArea=1000, 
                #filter = 0,
                #draw = False):
    imgBgr: np.ndarray = cv2.imread(str(Path(path)))
    #dim = rescaleImage(imgBgr)
    #imgBgr = cv2.resize(imgBgr, dim)
    imgHSV: np.ndarray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(src=imgHSV, lowerb=np.array([10, 0, 0]), upperb=np.array([175, 255, 60]))
    imgHSVModified: np.ndarray = cv2.bitwise_and(imgBgr, imgBgr, mask=mask)

    imgGray = cv2.cvtColor(imgHSVModified, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.bilateralFilter(imgGray, 5, 75, 75) #cv2.GaussianBlur(imgGray,(5,5),0)
    thresh, imgBit = cv2.threshold(imgBlur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarcy = cv2.findContours(imgBit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    imgBgrCopy = np.copy(imgBgr)
    for cnt in filter(lambda c: cv2.contourArea(c) > 10000, contours):
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        cv2.drawContours(imgBgrCopy, [np.int0(box)], -1, color = (0, 255, 0), thickness=2)

        for i in box:
            cv2.circle(imgBgrCopy, (int(i[0]), int(i[1])), 3, (0,0,255), -1)
        #show_img(imgBgrCopy)
        lengthInPixels = int(round(calculateDistance(box[0], box[1]) / 4, 0))
        widthInPixels  = int(round(calculateDistance(box[1], box[3]) / 4, 0))
        lengthInCm = round(calculateDistance(box[0], box[1]) / 10, 1)
        widthInCm = round(calculateDistance(box[1], box[3]) / 10, 1)

        point1_x = np.int0(box[0][0])
        point1_y = np.int0(box[0][1])

        point2_x = np.int0(box[1][0])
        point2_y = np.int0(box[1][1])

        point3_x = np.int0(box[3][0])
        point3_y = np.int0(box[3][1])

        cv2.arrowedLine(imgBgrCopy, 
                        (point1_x, point1_y), 
                        (point2_x, point2_y), 
                        (255, 0, 255), 8, 8, 0, 0.05)
        cv2.arrowedLine(imgBgrCopy, 
                        (point1_x, point1_y), 
                        (point3_x, point3_y), 
                        (255, 0, 255), 8, 8, 0, 0.05)
        cv2.putText(imgBgrCopy, 
                    '{}cm'.format(lengthInCm), 
                    (point1_x + lengthInPixels, point1_y - 20), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    5,
                    (255, 0, 255),
                    8)
        cv2.putText(imgBgrCopy, 
                    '{}cm'.format(widthInCm), 
                    (point1_x - widthInPixels, point3_y - 20), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    5,
                    (255, 0, 255),
                    8)

        #show_img(imgBgrCopy)

    #show_img(imgBgrCopy)
    return imgBgrCopy

def calculateDistance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    """ if __name__ == '__main__':
    getContours() """

    """ imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv2.erode(imgDial, kernel, iterations=2)
    
    if showCanny:cv2.imshow('Canny', imgThres)

    contours, hierarchy = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area  = cv2.contourArea(i)
        if area > minArea:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*perimeter, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])
    finalContours = sorted(finalContours, key = lambda x:x[1], reverse = True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0,0,255), 3)
    return img, finalContours



def warpImg (img, points, w, h):
    pass """