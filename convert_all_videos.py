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
    classifiers = np.array(["goodbye","hey","yes","no"])
    array_of_video_frames = np.array([],int)
    array_of_classifiers = np.array([])
    number_of_videos = 30
    for classifier in classifiers:
        for num in range(1,number_of_videos+1):
            array_of_classifiers = np.append(array_of_classifiers,classifier)
            for frame in range(1,21):
                file = "videos/"+classifier+str(num)+"/frame"+str(frame)+".jpg"
                try:
                    pimg = Image.open(file)
                except IOError:
                    break
                img_array = kpi.img_to_array(pimg)
                pimg.close()
                #print(img_array.shape)
                array_of_video_frames = np.append(array_of_video_frames,img_array)
    #array_of_classifiers = np.reshape(array_of_classifiers, (
    array_of_video_frames = np.reshape(array_of_video_frames, (number_of_videos,20,640,360,1))
    #array_of_classifiers = np.reshape(array_of_classifiers, (number_of_videos))
    return array_of_video_frames, array_of_classifiers
