import os
import imageio
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle
from main import readImages
dirname='annotated'

if dirname not in os.listdir(os.getcwd()):
    os.mkdir(dirname)
#images = readImages('img/')
#print(len(images))

foldernames = os.listdir('img/')
print(foldernames)
#images = [imageio.imread('img/'+fn+'/image0.jpg')[::,::].astype(np.float32)/255. for fn in filenames]#glob.glob(imgFolder+'*.jpg')]
###Image Annotation###
i=0
for f in foldernames:
    
    img = Image.open('img/'+f+'/image0.jpg')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r'./sans-serif.ttf', 28)
    draw.text((10,60),"Copyrighted",font=font)
    draw.text((40,100),"For Test",font=font)
    draw.text((20,140),"Purposes",font=font)
    img.save(os.path.join(dirname,'image_annotated_'+str(i+1)+'.png'))
    i+=1

###Image Cropping###
"""
for i in range(len(images)):
    rows,cols = images[i].shape
    
    img = images[i]
    draw = ImageDraw.Draw(img)
    #M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
    #dst = cv2.warpAffine(textImg,M,(cols,rows))
    #images[i]+=textImg
    imageio.imwrite(os.path.join(dirname,'image_annotated_'+str(i+1)+'.png'), img)
"""