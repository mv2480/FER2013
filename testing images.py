# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:02:03 2020

@author: 91989
"""
import cv2
import numpy as np

emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

img = cv2.imread('test.jpg')
img = cv2.resize(img,(48,48))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = np.reshape(img,[1,48,48,1])
classes = classifier.predict_classes(img)
print(emotion_map[int(classes)])
    
