import cv2
import cvzone
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

totalMoney = 0
def empty(a):
    pass

cv2.namedWindow("settings")
cv2.resizeWindow("settings",640,240)
cv2.createTrackbar("threshold1","settings",39,255,empty)
cv2.createTrackbar("threshold2","settings",168,255,empty)

def preProcessing(img):
    imgPre = cv2.GaussianBlur(img,(5,5),3)
    thresh1 = cv2.getTrackbarPos("threshold1","settings")
    thresh2 = cv2.getTrackbarPos("threshold2","settings")
    imgPre = cv2.Canny(imgPre,thresh1,thresh2)
    kernel = np.ones((3,3),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations=1)
    imgPre = cv2.morphologyEx(imgPre,cv2.MORPH_CLOSE, kernel)

    return imgPre

while True:
    success, img = cap.read()
    imgPre = preProcessing(img)

    imgContours, conFound = cvzone.findContours(img,imgPre,minArea=20)
    totalMoney = 0
    if conFound:
        for count,contour in enumerate(conFound):
            peri = cv2.arcLength(contour['cnt'],True)
            approx = cv2.approxPolyDP(contour['cnt'],0.02*peri,True)
            
            if len(approx)>3:
                area = (contour['area'])
               
                if area<1600:
                    totalMoney +=2
                elif 1600<area<2200:
                    totalMoney  +=1
                else:
                    totalMoney =+5
    
    print(totalMoney) 
    
    imgStacked = cvzone.stackImages([img,imgPre,imgContours],2,0.7)
    cvzone.putTextRect(imgStacked,f'Rs.{totalMoney}',(50,50))
    cv2.imshow("Image",imgStacked)
    
    cv2.waitKey(1)

