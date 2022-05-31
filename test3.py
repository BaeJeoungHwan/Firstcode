# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:51:22 2021

@author: n9310
"""

import cv2, os
import face_recognition as fr
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

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
#print(descs['SongjoongGi'])
#추출 벡터 개수 = 128개
#a = len(descs['SongjoongGi'])
#print(a)

img_bgr = cv2.imread('images/image_10.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

img_shapes2 = fr.face_locations(img_rgb)
descriptors = fr.face_encodings(img_rgb, img_shapes2)

fig, ax = plt.subplots(1, figsize=(20,20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):
    
    Found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1)
        print(dist)
        if dist < 0.6:
            Found = True
            print('good')
            #text = ax.text(rects[i][0][0], rects[i][0][1], name,
           #                color='b', fontsize=40, fontweight='bold')
          #  test.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects])
        #    rect = patches.Rectangle(rects[i][0][1],
#                                     rects[i][1][1] - rects[i][0][1],
                        #             rects[i][1][0] - rects[i][0][0],
                         #            linewidth=2, edgecolor='w',facecolor='none')
          #  ax.add_patch(rect)
            
            break
        
        if not Found:
            #text = ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
             #       color='r', fontsize=20, fontweight='bold')
          #  test.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects])
          #  rect = patches.Rectangle(rects[i][0],
         #                            rects[i][1][1] - rects[i][0][1],
          #                           rects[i][1][0] - rects[i][0][0],
           #                          linewidth=2, edgecolor='r',facecolor='none')
            #ax.add_patch(rect)
            print('bad')
             
            
plt.show()
            
            
            
            
            
            
            
            