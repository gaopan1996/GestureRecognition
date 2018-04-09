import numpy as np
import cv2
import os
import argparse

classifier = "goodbye"
array_of_video_frames = np.array([],int)
for num in range(1,31):
    for frame in range(1,21):
        file = "videos/"+str(classifier)+str(num)+"/frame"+str(frame)+".jpg")
        img_array = cv2.imread(file,0)
        array_of_video_frames = np.append(array_of_video_frames,img_array)
        h,w = img_array.shape
        
array_of_video_frames = np.reshape(array_of_video_frames, (30,20,640,360))
