import os
import imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
from main import readImages
dirname='annotated'
images = readImages('img/')
print(len(images))

if dirname not in os.listdir(os.getcwd()):
    os.mkdir(dirname)
    
for i in range(len(images)):
    rows,cols = images[i].shape
    overlay = np.copy(images[i])
    output = np.copy(images[i])
    alpha = 0.8
    #textImg = np.zeros((rows,cols),np.uint8)
    cv2.putText(overlay, "copyrighted", (0, cols//2), cv2.FONT_HERSHEY_SIMPLEX,0.85, (250,250,250),2,cv2.LINE_AA)
    cv2.addWeighted(overlay,alpha,output,1-alpha,0,output)
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    #dst = cv2.warpAffine(textImg,M,(cols,rows))
    #images[i]+=textImg
    imageio.imwrite(os.path.join(dirname,'image_annotated_'+str(i+1)+'.png'), output.astype(np.uint8))
    