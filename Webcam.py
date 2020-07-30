# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:00:50 2020

@author: 91989
"""


emotion_map = {0: 'Angry', 1: 'Digust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

vid = cv2.VideoCapture(0)
while(True): 
    ret, img = vid.read() 
    img = cv2.resize(img,(48,48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img,[1,48,48,1])
    classes = classifier.predict_classes(img)
    print(emotion_map[int(classes)])
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()