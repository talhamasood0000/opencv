# Virtual Painter

from cv2 import cv2
import numpy as np

frameWidth=640
frameHeight=480
cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,130)


def findColor(img,myColors,myColorValue):
    imgHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    count=0
    newPoints=[]
    for color in myColors:
        lower=np.array(color[0:3])
        upper=np.array(color[3:6])
        mask=cv2.inRange(imgHSV,lower,upper)
        cv2.imshow(str(color[0]),mask)
        x,y=getContours(mask)
        x=int(x)
        y=int(y)
        cv2.circle(imgResult,(x,y),10,myColorValue[count],cv2.FILLED)
        if x!=0 and y!=0:
            newPoints.append([x,y,count])
        count=count+1
    return newPoints



def getContours(img):
    countours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#retrieve the extreme outer contours
    x,y,w,h=0,0,0,0
    for cnt in countours:
        area=cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgResult,cnt,-1,(255,0,0),3)#image on which to draw, contour to draw, to draw all(-1), colour, thickness
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            x,y,w,h=cv2.boundingRect(approx)
    return x+w/2,y

def drawOnCanvas(myPoints,myColorValue):
    for point in myPoints:
        cv2.circle(imgResult,(point[0],point[1]),10,myColorValue[point[2]],cv2.FILLED)



myColors=[[5,107,0,19,255,255],#list of colors we want to detect, first is orange
            [133,56,0,169,156,255],
            [57,76,0,100,255,255]]
myColorValue=[[51,153,255], #BGR not RGB
                [255,0,255],
                [0,255,0]]


myPoints=[] #[x,y,colorID]


while True:
    success,img=cap.read()
    imgResult=img.copy()
    newPoints=findColor(img,myColors,myColorValue)
    if len(newPoints)!=0:
        for newP in newPoints:
            myPoints.append(newP)
    if len(myPoints)!=0:
        drawOnCanvas(myPoints,myColorValue)
    cv2.imshow("Result",imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    



