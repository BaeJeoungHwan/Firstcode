# -*- coding: utf-8 -*-
# 참고 사이트 https://jngmk.netlify.app/dev/python/2020-03-19-face-recognition-with-openCV-and-dlib
"""
Created on Tue Mar 23 02:57:09 2021

@author: n9310
"""
# Face Detection
import cv2
import face_recognition as fr
from IPython.display import Image, display
from matplotlib import pyplot as plt
import numpy as np
#import matplotlib.pylab as plt
img_path = {
    'SongjoongGi' : 'images/image_5.jpg',
    'JeonyeoBin' : 'images/image_6.jpg'
    }

descs = {
    'SongjoongGi' : None,
    'JeonyeoBin' : None
    }

for name, img_path in img_path.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_shapes = fr.face_locations(img_rgb)
    descs[name] = fr.face_encodings(img_rgb, img_shapes)[0]

np.save('images/descs.npy',descs)


image = fr.load_image_file('images/image_2.jpg')
face_locations = fr.face_locations(image)

for (top, right, bottom, left) in face_locations:
  cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
   # (그릴 곳, 시작점, 끝점, 색, 두께)
   
spot = (right-left)
spot2 = (top-bottom)
cv2.putText(image, 'test', (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
  
enc_face = fr.face_encodings(image)
#print(enc_face)


#bgr을 rgb로 변경
#b, g, r = cv2.split(image)
#image2 = cv2.merge([r, g, b])

#cv2.imshow("Image",image2)
#cv2.waitKey(0)

plt.rcParams['figure.figsize'] = (16, 16)
plt.imshow(image)
plt.show()
