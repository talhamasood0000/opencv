# (Pixels)
# Vga 640x480
# 1280x720
# 4k=3840x2160
# Binary Image (Black=0,White=1)
# 8 Bit value gives us 256 cshades of black and white

# Chapter1 Read images Videos Webcam
from cv2 import cv2
import numpy as np
# image read
# img=cv2.imread(r'C:\Users\Talha Masood\Desktop\ajmair.png')
# cv2.imshow('output window',img)
# cv2.waitKey(0) #milliseconds, 0 means infinite

# videoread
# cap=cv2.VideoCapture(r'C:\Users\Talha Masood\Desktop\Lec6_part1.mp4')

# while True:
    # success,img=cap.read() #success is either true or false
    # cv2.imshow('Video',img)
    # if cv2.waitKey(1) & 0xFF==ord('q'):
        # break

# webcam
# cap=cv2.VideoCapture(0) #0 means webcam
# cap.set(3,640) #id 3 is width of the screen
# cap.set(4,480) #id 4 is height of the screen
# cap.set(10,100) #id 10 is brightness of the screen

# while True:
#     success,img=cap.read() #success is either true or false
#     cv2.imshow('Video',img)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break


# Chapter2
# to convert into gray
# img=cv2.imread(r'C:\Users\Talha Masood\Desktop\ajmair.png')
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # in cv2 its bgr not rgb
# cv2.imshow('Gray Image',imgGray)
# cv2.waitKey(0)

# to blur the image
# imgBlur=cv2.GaussianBlur(imgGray,(7,7),0) # kernal size(always odd), sigma=0
# cv2.imshow('Blur Image',imgBlur)
# cv2.waitKey(0)

# canny edge detector
# imgCanny=cv2.Canny(img,100,100) #threshold1=100,threshold2=100 
# cv2.imshow('Canny Image',imgCanny)
# cv2.waitKey(0)


# Dilation(increase the thickness of edges)
# kernal=np.ones((5,5),np.uint8) #unit8 means values range from 0-255
# imgDilation=cv2.dilate(imgCanny,kernal,iterations=1) #kernal matrix, iterations means the thickness
# cv2.imshow('Dilation Image',imgDilation)
# cv2.waitKey(0)

# Erosion(Opposite of dilation)
# imgErode=cv2.erode(imgDilation,kernal,iterations=1)
# cv2.imshow('Erode Image',imgErode)
# cv2.waitKey(0)

# Chapter 3 Resizing and Cropping
# +ve x-axix is towards east, +ve y-axix is towards South (By openCV convention)
# img=cv2.imread(r'C:\Users\Talha Masood\Desktop\abc.jpg')

# Resize the image
# imS = cv2.resize(img, (500, 500)) #Width and then Height
# print(img.shape)
# cv2.imshow('output window',imS)
# cv2.waitKey(0)

# Crop Image
# imgCrop=img[0:1000,200:1000] #Height and then Width
# cv2.imshow('output window',imgCrop)
# cv2.waitKey(0)


# Chapter 4 Shapes and texts

# img=np.zeros((512,512,3),np.uint8)
# img[200:300,100:300]=255,0,0 #Blue Color for some part
# img[:]=255,0,0 #all black

# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3) # starting point, ending point, color(green), thickness
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED)
# cv2.circle(img,(400,50),30,(0,255,255),5) #startting point, radius, color, thickness
# cv2.putText(img,"OPEN CV",(300,100),cv2.FONT_HERSHEY_TRIPLEX,1,(0,150,0),1) # text, starting point, font style, scale, color, thickness

# cv2.imshow("Image",img)
# cv2.waitKey(0)

# Chapter 5 Wrap Prespective

# width,height=250,350
# img=cv2.imread(r'C:\Users\Talha Masood\Desktop\abc.jpg')
# pts1=np.float32([[317,309],[1390,237],[273,1433],[1377,1449]]) #add points
# pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])

# matrix=cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput=cv2.warpPerspective(img,matrix,(width,height))
# cv2.imshow('output window',imgOutput)
# cv2.waitKey(0)

#Chapter 6 Joining Images

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# img=cv2.imread(r'C:\Users\Talha Masood\Desktop\abc.jpg')
# imS = cv2.resize(img, (500, 500))
# imver=np.vstack((imS,imS)) #vertical stacking of images
# imhor=np.hstack((imS,imS)) #horizontal stacking of images

# imgStack=stackImages(0.5,([imS,imS,imS],[imS,imS,imS]))
# cv2.imshow('output window',imgStack)
# cv2.waitKey(0)


# Chapter 7 Color Detection

def empty(a):
    pass

# cv2.namedWindow("TrackBars") #to make tracker(value changer)
cv2.resizeWindow("TrackBars",640,300)

cv2.createTrackbar("Hue Min","TrackBars",0,179,empty) #name, to which we want to apply, minimum value,maximum value(of hue, in our case it is total 180 )
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Saturation Min","TrackBars",0,255,empty)
cv2.createTrackbar("Saturation Max","TrackBars",255,255,empty)
cv2.createTrackbar("Value Min","TrackBars",0,255,empty)
cv2.createTrackbar("Value Max","TrackBars",255,255,empty)


while True:
    img=cv2.imread(r'C:\Users\Talha Masood\Desktop\abc.jpg')
    imS = cv2.resize(img, (500, 500))
    imgHSV=cv2.cvtColor(imS,cv2.COLOR_BGR2HSV) #convert it into HSV colors
    
    h_min=cv2.getTrackbarPos("Hue Min","TrackBars") #to read values from the tracker
    h_max=cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min=cv2.getTrackbarPos("Saturation Min","TrackBars")
    s_max=cv2.getTrackbarPos("Saturation Max","TrackBars")
    v_min=cv2.getTrackbarPos("Value Min","TrackBars")
    v_max=cv2.getTrackbarPos("Value Max","TrackBars")

    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])

    mask=cv2.inRange(imgHSV,lower,upper)

    imageResult=cv2.bitwise_and(imS,imS,mask=mask)
     #adding the images(when both the pixels are present it will add them)


    # cv2.imshow('output window',mask)
    # cv2.imshow('out', imageResult)

    imgStack=stackImages(0.6,([imS,imgHSV],[mask,imageResult]))
    # cv2.imshow('out', imgStack)
    # cv2.waitKey(1)
    break

#Chapter 8 Contours/ Shape Detection

#function to get contours
def getContours(img):
    countours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#retrieve the extreme outer contours
    for cnt in countours:
        area=cv2.contourArea(cnt)
        
        if area>500:
            cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)#image on which to draw, contour to draw, to draw all(-1), colour, thickness
            peri=cv2.arcLength(cnt,True)
            
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)

            objCor=len(approx)
            x,y,w,h=cv2.boundingRect(approx)
            if objCor==4:
                aspRat=w/float(h)
                if aspRat>0.99 and aspRat<1.01:
                    ObjectType="Square"
                else:
                    ObjectType="Rectangle"
                
            elif objCor==4:
                ObjectType="Triangle"
            elif objCor>4:
                ObjectType="Circle"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,132))# bounding boxex around the shape
            cv2.putText(imgContour,ObjectType,
                (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,(0,0,0),2)



img=cv2.imread(r'C:\Users\Talha Masood\Desktop\shap.jpg')
imS = cv2.resize(img, (500, 500))

imgContour=imS.copy()

imgGray=cv2.cvtColor(imS,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
imgCanny=cv2.Canny(imgBlur,50,50)

getContours(imgCanny)

imgBlank=np.zeros_like(imS)
imgStack=stackImages(0.6,([imS,imgGray,imgBlur],[imgCanny,imgContour,imgBlank]))

# cv2.imshow('out', imgStack)
# cv2.waitKey(0)



# Chapter 9 Facial Detection
# Viola and Jones Method
# opencv cascades(predefined functions to detect many things in opencv)



faceCascade=cv2.CascadeClassifier(r"C:\Users\Talha Masood\AppData\Local\Programs\Python\Python39\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
img=cv2.imread(r'C:\Users\Talha Masood\Desktop\abc.jpg')
imS = cv2.resize(img, (500, 500))
imgGray=cv2.cvtColor(imS,cv2.COLOR_BGR2GRAY)



faces=faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(imS,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow('out', imS)
cv2.waitKey(0)

