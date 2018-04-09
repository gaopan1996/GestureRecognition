import numpy as np
import cv2
import os
def img_to_array(file):
    img = cv2.imread(file,-1)
    return img
classifier = "goodbye"
goodbye = np.array([])
for num in range(0,30):
    for frame in range(0,20):
        img_array = img_to_array("videos/"+str(classifier)+str(num)+"/frame"+str(frame)+".jpg")
        print("videos/"+str(classifier)+str(num)+"/frame"+str(frame)+".jpg")
        np.append(goodbye,img_array)
np.savetxt('test.txt', goodbye)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
