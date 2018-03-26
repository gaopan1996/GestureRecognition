import numpy as np
import cv2
import time

max_area = 0
val = float(input("Enter value\n"))
blur_val = int(input("Enter blur\n"))
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gesture_cascade = cv2.CascadeClassifier('haarcascade_gesture.xml')
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

#Decrease frame size
cap.set(3, 1000)
cap.set(4, 600)
#cv2.namedWindow('HSV_TrackBar')
#h,s,v = 100,100,100
#cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
#cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
#cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    gestures = gesture_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in gestures:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

#    for (x,y,w,h) in gestures:
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]

    blur = cv2.GaussianBlur(gray, (blur_val,blur_val), 0)
    ret,thresh1 = cv2.threshold(blur,val,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnt=contours[0]
    for i in range(len(contours)):
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
            cnt=contours[ci]
    hull = cv2.convexHull(cnt)
    print("cnt: ")
    print(cnt)
    print("hull: ")
    print(hull)
    drawing = np.zeros(frame.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),3)
    cv2.drawContours(drawing,[hull],0,(0,0,255),3)
    cv2.drawContours(frame,[cnt],0,(0,255,0),3)
    cv2.drawContours(frame,[hull],0,(0,0,255),3)
#    mask = cv2.inRange(gray, 0,255)
#    mask = cv2.erode(mask, skinkernel, iterations = 1)
#    mask = cv2.dilate(mask, skinkernel, iterations = 1)


#    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#    ret, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    fgmask = fgbg.apply(thresh1)
    #h = cv2.getTrackbarPos('h','HSV_TrackBar')
    #s = cv2.getTrackbarPos('s','HSV_TrackBar')
    #v = cv2.getTrackbarPos('v','HSV_TrackBar')
    #print(h,s,v)
    # Our operations on the frame come here
    # Display the resulting frame
    cv2.imshow('blur', blur)
    cv2.imshow('frame',thresh1)
    cv2.imshow('actual',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
