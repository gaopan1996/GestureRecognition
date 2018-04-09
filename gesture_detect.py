import cv2
import numpy as np
import codecs, json
import time
import argparse
import os
def nothing(x):
    pass

cv2.namedWindow('Variable Values')
cv2.createTrackbar('blur', 'Variable Values',11,179,nothing)
cv2.createTrackbar('wait', 'Variable Values', 25, 100, nothing)
cv2.createTrackbar('mask', 'Variable Values', 20, 255, nothing)
cv2.createTrackbar('mask_lower', 'Variable Values',0, 255, nothing)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
blur_val = 10
wait = 25
mask_hsv = 20
mask_lower = 0

if not os.path.exists("videos"):
    os.makedirs("videos")


data = open("output.dat", "w")
open('data.json','w')
cycle = 1
calibrated_x = calibrated_y = calibrated_h = calibrated_w = 0
parser = argparse.ArgumentParser()
filename = None
parser.add_argument('-f', '--file', dest="file", default="")
options = parser.parse_args()
if options.file:
   filename = options.file

if filename is None:
    cap = cv2.VideoCapture(0)
    #Set frame size and initial values
    cap.set(3, 1280)
    cap.set(4, 1024)
    while(True):
        start_time = time.time()
        #data.write("Cycle " + str(cycle) + "\n")
        jsonobj = {}
        jsonobj[str(cycle)] = { 'face_center_coordinate':[], 'hand_center_coordinate': [], 'finger_points_coordinate': [], 'finger_distance_from_center_palm': [], 'hull_coordinates':[], 'hull_defects_coordinates':[], 'face_bounding_rectangle': [], 'contour_coordinates': [] }
        #Capture frames from the camera
        ret, frame = cap.read()
        hand = frame
        #Use Haar Cascade to detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        #face_json = {'face_center_coordinate': []}
        #Draw the region of interest 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            center_face = (int(x+w/2),int(y+h/2))
            center_face_str = np.array(map(str, center_face))
            #data.write("Face center: ("+ center_face_str[0]+", "+center_face_str[1]+")\n")
            #Draw center of face
            cv2.circle(frame, center_face, 7, [100,0,255], 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, 'Face Center', center_face , font, 2, (255,255,255), 2)
            #face_json = {'face_center_coordinate':[ x+w/2, y+h/2 ]}
            jsonobj[str(cycle)]['face_center_coordinate'].append("("+str(center_face[0])+ ", "+str(center_face[1])+")")
        #jsonobj[str(cycle)]['face_center_coordinate'].append("("+center_face_str[0]+ ", "+center_face_str[1]+")")
        #file_path = "path1.json" ## your path variable
        #json.dump(face_json, codecs.open(file_path, 'a', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        blur_val = cv2.getTrackbarPos('blur', 'Variable Values')
        #Blur the image to remove noised
        blur = cv2.blur(frame,(blur_val,blur_val))
        #blur = cv2.GaussianBlur(frame, (blur_val,blur_val), 0)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", hsv)
        #Mask blurred image with Hue values of 0-20.
        mask_hsv = cv2.getTrackbarPos('mask','Variable Values')
        mask_lower = cv2.getTrackbarPos('mask_lower','Variable Values')
        mask = cv2.inRange(hsv,np.array([mask_lower,50,50]),np.array([mask_hsv,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Morph the image to get rid of noise.
        #Dilation increases the noise while erosion erases noise that was enlarged.
        dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1) 
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        cv2.imshow("test",thresh)
        #Find contours of the filtered frame
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Find largest contour on the frame
        max_area=100
        i=0	
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                i = n  
        if len(contours) > 0:        
            cnts = contours[i]
        else:
            continue
        for i in range(cnts.shape[0]):
            #coord = tuple(cnts[i][0])
            jsonobj[str(cycle)]['contour_coordinates'].append("(" + str(cnts[i][0][0]) + ", " + str(cnts[i][0][1]) + ")")
        #contour_json = {'contours': cnts.tolist()}
        # json.dump(contour_json, codecs.open('data.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #Find convex hull of hand contour
        hull = cv2.convexHull(cnts)
        
        #Find convex defects of the hand
        hull2 = cv2.convexHull(cnts, returnPoints = False)
        defects = cv2.convexityDefects(cnts, hull2)
        
        #Get defect points and draw them in the frame
        FarDefect = []
        temp = ''
        temp1 = ''
        temp2 = ''
        #hull_json = {'hull_coordinates': []}
        defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            cv2.circle(frame,far,5,[150,255,255],2)
            jsonobj[str(cycle)]["hull_defects_coordinates"].append("(" + str(far[0]) + ", " + str(far[1]) + ")")
            jsonobj[str(cycle)]["hull_coordinates"].append("(" + str(start[0]) + ", " + str(start[1]) + ")")
            temp = temp + "(" + str(far[0]) + ", " + str(far[1]) + ") "
            temp1 = temp1 + "(" + str(start[0]) + ", " + str(start[1]) + ") "
            temp2 = temp2 + "(" + str(end[0]) + ", " + str(end[1]) + ") "
        #temp = temp + "]"
        #temp1 = temp1 + "]"
        #temp2 = temp2 + "]"
    
        #data.write("Raw Hull defects: " + temp + "\n")
        #data.write("Hull points: " + temp1 + "\n")
        #data.write("Hull end points: " + temp2 + "\n")
        #Find center of hand
        moments = cv2.moments(cnts)
        if moments['m00']!=0:
            x = int(moments['m10']/moments['m00'])
            y = int(moments['m01']/moments['m00'])
        center=(x,y)    
        #data.write("Center of hand: " + "(" + str(x) + ", " + str(y) + ")\n")
        #Draw center mass
        cv2.circle(frame,center,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,'Center',tuple(center),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            center = np.array(center)
            distance = np.sqrt(np.power(x[0]-center[0],2)+np.power(x[1]-center[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        #data.write("Finger points: ")
        for i in range(0,len(fingers)):
            #data.write("(" + str(fingers[i][0]) + ", " + str(fingers[i][1]) + ") ")
            jsonobj[str(cycle)]["finger_points_coordinate"].append ("(" + str(fingers[i][0]) + ", " + str(fingers[i][1]) + ")")
            distance = np.sqrt(np.power(fingers[i][0]-center[0],2)+np.power(fingers[i][1]-center[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        #data.write("\nFinger distance from center of palm: ")
        for i in range(0,len(fingers)):
            #data.write(str(fingerDistance[i]) + ", ")
            jsonobj[str(cycle)]["finger_distance_from_center_palm"].append(str(fingerDistance[i]))
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1
        #data.write("\n")
        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        if cycle == 1:
            calibrated_x = x
            calibrated_y = y
            calibrated_h = h
            calibrated_w = w
        #data.write("Bounding Rectangle: (" + str(x) + "," + str(y) + ") (" + str(x+w) + "," +str(y) +") (" + str(x) + "," + str(y+h) + ") (" + str(x+w) + ","+str(y+h)+")\n")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x) + "," + str(y) + ")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x+w) + "," +str(y) +")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x) + "," + str(y+h)+")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x+w) + ","+str(y+h)+")")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,255,255),2)
        
        ##### Show final image ########
        cv2.imshow('Final',frame)
        cv2.imshow('Hand', hand[calibrated_y:calibrated_y+calibrated_h, calibrated_x:calibrated_x+calibrated_w])
        cv2.imshow('Hand_Mask', mask[calibrated_y:calibrated_y+calibrated_h, calibrated_x:calibrated_x+calibrated_w])
#        cv2.imshow('Hand', hand[x:x+w,y:y+h])
        elapsed_time = time.time() - start_time 
        json.dump(jsonobj, codecs.open('data.json', 'a', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #data.write("Elapsed Time: " + str(elapsed_time) + "\n")
        print("Elapsed Time: " + str(elapsed_time) + "\n")
        cycle = cycle + 1
        #close the output video by pressing 'ESC'
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
else:
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    #out = cv2.VideoWriter("videos/out.avi",fourcc, 20.0, (720,480))
    cap = cv2.VideoCapture(filename)
    
    while(cap.isOpened()):
        start_time = time.time()
        #data.write("Cycle " + str(cycle) + "\n")
        jsonobj = {}
        jsonobj[str(cycle)] = { 'face_center_coordinate':[], 'hand_center_coordinate': [], 'finger_points_coordinate': [], 'finger_distance_from_center_palm': [], 'hull_coordinates':[], 'hull_defects_coordinates':[], 'face_bounding_rectangle': [], 'contour_coordinates': [] }
        #Capture frames from the camera
        ret, frame = cap.read()
        if not ret:
           break
        hand = frame
        #Use Haar Cascade to detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #    face_json = {'face_center_coordinate': []}
        #Draw the region of interest 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            center_face = (int(x+w/2),int(y+h/2))
            center_face_str = np.array(map(str, center_face))
            #data.write("Face center: ("+ center_face_str[0]+", "+center_face_str[1]+")\n")
            #Draw center of face
            cv2.circle(frame, center_face, 7, [100,0,255], 2)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, 'Face Center', center_face, font, 2, (255,255,255), 2)
    #        face_json = {'face_center_coordinate':[ x+w/2, y+h/2 ]}
            jsonobj[str(cycle)]['face_center_coordinate'].append("("+str(center_face[0])+ ", "+str(center_face[1])+")")
    #    jsonobj[str(cycle)]['face_center_coordinate'].append("("+center_face_str[0]+ ", "+center_face_str[1]+")")
    #    file_path = "path1.json" ## your path variable
    #    json.dump(face_json, codecs.open(file_path, 'a', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        blur_val = cv2.getTrackbarPos('blur', 'Variable Values')   
        #Blur the image to remove noised
        blur = cv2.blur(frame,(blur_val,blur_val))
    #   blur = cv2.GaussianBlur(frame, (blur_val,blur_val), 0)
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", hsv)
        #Mask blurred image with Hue values of 0-20.
        mask_hsv = cv2.getTrackbarPos('mask','Variable Values')
        lower_mask = cv2.getTrackbarPos('mask_lower','Variable Values')
        mask = cv2.inRange(hsv,np.array([lower_mask,50,50]),np.array([mask_hsv,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Morph the image to get rid of noise.
        #Dilation increases the noise while erosion erases noise that was enlarged.
        dilation = cv2.dilate(mask,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1) 
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        ret,thresh = cv2.threshold(median,127,255,0)
        cv2.imshow("test",thresh)
        
        #Find contours of the filtered frame
        _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
        
        #Find largest contour on the frame
        max_area=100
        i=0	
        for n in range(len(contours)):
            cnt = contours[n]
            area = cv2.contourArea(cnt)
            if(area > max_area):
                max_area = area
                i = n  
                
        if len(contours) > 0:        
            cnts = contours[i]
        else:
            continue

        for i in range(cnts.shape[0]):
    #        coord = tuple(cnts[i][0])
            jsonobj[str(cycle)]['contour_coordinates'].append("(" + str(cnts[i][0][0]) + ", " + str(cnts[i][0][1]) + ")")
    #    contour_json = {'contours': cnts.tolist()}
    #    json.dump(contour_json, codecs.open('data.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #Find convex hull of hand contour
        hull = cv2.convexHull(cnts)
        
        #Find convex defects of the hand
        hull2 = cv2.convexHull(cnts, returnPoints = False)
        defects = cv2.convexityDefects(cnts, hull2)
        
        #Get defect points and draw them in the frame
        FarDefect = []
        temp = ''
        temp1 = ''
        temp2 = ''
    #    hull_json = {'hull_coordinates': []}
        defects
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            end = tuple(cnts[e][0])
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            cv2.circle(frame,far,5,[150,255,255],2)
            jsonobj[str(cycle)]["hull_defects_coordinates"].append("(" + str(far[0]) + ", " + str(far[1]) + ")")
            jsonobj[str(cycle)]["hull_coordinates"].append("(" + str(start[0]) + ", " + str(start[1]) + ")")
            temp = temp + "(" + str(far[0]) + ", " + str(far[1]) + ") "
            temp1 = temp1 + "(" + str(start[0]) + ", " + str(start[1]) + ") "
            temp2 = temp2 + "(" + str(end[0]) + ", " + str(end[1]) + ") "
    #    temp = temp + "]"
    #    temp1 = temp1 + "]"
    #    temp2 = temp2 + "]"
    
        #data.write("Raw Hull defects: " + temp + "\n")
        #data.write("Hull points: " + temp1 + "\n")
        #data.write("Hull end points: " + temp2 + "\n")
        #Find center of hand
        moments = cv2.moments(cnts)
        if moments['m00']!=0:
            x = int(moments['m10']/moments['m00'])
            y = int(moments['m01']/moments['m00'])
        center=(x,y)    
        #data.write("Center of hand: " + "(" + str(x) + ", " + str(y) + ")\n")
        #Draw center mass
        cv2.circle(frame,center,7,[100,0,255],2)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(frame,'Center',tuple(center),font,2,(255,255,255),2)     
        
        #Distance from each finger defect(finger webbing) to the center
        distanceBetweenDefectsToCenter = []
        for i in range(0,len(FarDefect)):
            x =  np.array(FarDefect[i])
            center = np.array(center)
            distance = np.sqrt(np.power(x[0]-center[0],2)+np.power(x[1]-center[1],2))
            distanceBetweenDefectsToCenter.append(distance)
        
        #Get an average of three shortest distances from finger webbing to center mass
        sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
        AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
     
        #Get fingertip points from contour hull
        #If points are in proximity of 80 pixels, consider as a single point in the group
        finger = []
        for i in range(0,len(hull)-1):
            if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                if hull[i][0][1] <500 :
                    finger.append(hull[i][0])
        
        #The fingertip points are 5 hull points with largest y coordinates  
        finger =  sorted(finger,key=lambda x: x[1])   
        fingers = finger[0:5]
        #Calculate distance of each finger tip to the center mass
        fingerDistance = []
        #data.write("Finger points: ")
        for i in range(0,len(fingers)):
            #data.write("(" + str(fingers[i][0]) + ", " + str(fingers[i][1]) + ") ")
            jsonobj[str(cycle)]["finger_points_coordinate"].append ("(" + str(fingers[i][0]) + ", " + str(fingers[i][1]) + ")")
            distance = np.sqrt(np.power(fingers[i][0]-center[0],2)+np.power(fingers[i][1]-center[0],2))
            fingerDistance.append(distance)
        
        #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
        #than the distance of average finger webbing to center mass by 130 pixels
        result = 0
        #data.write("\nFinger distance from center of palm: ")
        for i in range(0,len(fingers)):
            #data.write(str(fingerDistance[i]) + ", ")
            jsonobj[str(cycle)]["finger_distance_from_center_palm"].append(str(fingerDistance[i]))
            if fingerDistance[i] > AverageDefectDistance+130:
                result = result +1
        #data.write("\n")
        #Print bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts)
        if cycle == 1:
            calibrated_x = x
            calibrated_y = y
            calibrated_h = h
            calibrated_w = w
        #data.write("Bounding Rectangle: (" + str(x) + "," + str(y) + ") (" + str(x+w) + "," +str(y) +") (" + str(x) + "," + str(y+h) + ") (" + str(x+w) + ","+str(y+h)+")\n")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x) + "," + str(y) + ")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x+w) + "," +str(y) +")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x) + "," + str(y+h)+")")
        jsonobj[str(cycle)]["face_bounding_rectangle"].append("(" + str(x+w) + ","+str(y+h)+")")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawContours(frame,[hull],-1,(255,255,255),2)
        
        ##### Show final image ########
        cv2.imshow('Final',frame)
        #cv2.imshow('Hand', hand[calibrated_y:calibrated_y+calibrated_h, calibrated_x:calibrated_x+calibrated_w])
        filename_exploded = filename.split(".")[0].split("/")
        outfilename = filename_exploded[len(filename_exploded)-1]
        #print(mask[calibrated_y:calibrated_y+510, calibrated_x:calibrated_x+310].shape)
        #out.write(mask)
        cv2.imshow('Hand_Mask', mask[calibrated_y:calibrated_y+640, calibrated_x:calibrated_x+360])
        #cv2.imshow('Hand_Mask', mask[calibrated_y:calibrated_y+510, calibrated_x:calibrated_x+310])
        if not os.path.exists("videos/"+outfilename):
            os.makedirs("videos/"+outfilename)
        if calibrated_y+640 > mask.shape[0]:
            calibrated_y = mask.shape[0]-640 
        cv2.imwrite("videos/"+outfilename+"/frame"+str(cycle)+".jpg",  mask[calibrated_y:calibrated_y+640, calibrated_x:calibrated_x+360])
        #print(mask.shape)
        #handjson = {'frame'+str(cycle): mask[calibrated_y:calibrated_y+510, calibrated_x:calibrated_x+310].tolist()}
        #cv2.imshow('Hand_Mask', mask[calibrated_y:calibrated_y+calibrated_h, calibrated_x:calibrated_x+calibrated_w])
        elapsed_time = time.time() - start_time 
        #json.dump(jsonobj, codecs.open('data.json', 'a', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #json.dump(handjson, codecs.open('hand.json', 'a', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
        #data.write("Elapsed Time: " + str(elapsed_time) + "\n")
        print("Elapsed Time: " + str(elapsed_time) + "\n")
        cycle = cycle + 1
        #close the output video by pressing 'ESC'
        wait = cv2.getTrackbarPos('wait','Variable Values')
        k = cv2.waitKey(wait) & 0xFF
        if k == 27:
            break
        if k == ord('p'):
            while True:
                k2 = cv2.waitKey(1) or 0xff
                cv2.imshow('Final', frame)
                if k2 == ord('p'):
                    break
            cv2.imshow('Final', frame)
           

cap.release()
cv2.destroyAllWindows()
