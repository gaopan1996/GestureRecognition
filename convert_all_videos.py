import numpy as np
import cv2
import os
import argparse
import keras.preprocessing.image as kpi
from PIL import Image
def convert_all():
    classifier = "goodbye"
    array_of_video_frames = np.array([],int)
    for num in range(1,31):
        for frame in range(1,21):
            file = "videos/"+str(classifier)+str(num)+"/frame"+str(frame)+".jpg"
            #print(file)
            img_array = cv2.imread(file,0)
            array_of_video_frames = np.append(array_of_video_frames,img_array)
            h,w = img_array.shape
            
    array_of_video_frames = np.reshape(array_of_video_frames, (30,20,640,360))
    return array_of_video_frames
def convert_all_keras():
    classifier = "goodbye"
    array_of_video_frames = np.array([],int)
    for num in range(1,31):
        for frame in range(1,21):
            file = "videos/"+str(classifier)+str(num)+"/frame"+str(frame)+".jpg"
            pimg = Image.open(file)
            img_array = kpi.img_to_array(pimg)
            #print(img_array.shape)
            array_of_video_frames = np.append(array_of_video_frames,img_array)
            
    array_of_video_frames = np.reshape(array_of_video_frames, (30,20,640,360))
    return array_of_video_frames

convert_all_keras()
