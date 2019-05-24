import os
import imageio
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pickle

DEFAULT_BASELINE_DCT = "baseline_dct.pkl"
DEFAULT_BLURRED_DCT = "gb_dct.pkl"

#Calculate DCT freqeuncy coefficients
"""
          N-1
y[k] = 2* sum x[n]*cos(pi*(2k+1)*(2n+1)/(4*N)), 0 <= k < N.
          n=0
"""

def readImages(imgFolder='img/'):
    """read all images in a given folder"""
    #Each image in images is a numpy array of shape 192x168(x1) (heightxwidth)
    #images datatype is a regular numpy list
    filenames = os.listdir(imgFolder)
    images = [imageio.imread('img/'+fn+'/image0.jpg')[::,::].astype(np.float32)/255. for fn in filenames]#glob.glob(imgFolder+'*.jpg')]
    return images

def gaussianBlur(img,ksize=(5,5),sigma=10):
    #kernel = cv2.getGaussianKernel(ksize,sigma)
    dst = np.zeros_like(img)
    cv2.GaussianBlur(src=img,dst=dst,ksize=ksize,sigmaX=0)
    return dst

def plotFace(original,blurred):
    plt.subplot(121),plt.imshow(original,cmap=cm.Greys_r),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blurred,cmap=cm.Greys_r),plt.title('Gaussian Blurred')
    plt.xticks([]), plt.yticks([])
    return None
"""
# Load data (deserialize)
with open('filename.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)
"""
if __name__=="__main__":
    images = readImages('img/')
    print('Found images:',len(images))
    baseline_dict,blurred_dict={},{}
    
    dirnames=['base_out','blurred_out']
    for d in dirnames:
        if d not in os.listdir(os.getcwd()):
            os.mkdir(d) 
            print('Creating output folder:',d)
    
    for i in range(len(images)):
        plt.figure(i)
        blurredImage = gaussianBlur(images[i])
        plotFace(images[i],blurredImage)
        baselineDCT, blurredDCT= cv2.dct(images[i]),cv2.dct(blurredImage)
        baseline_dict[i]=baselineDCT
        blurred_dict[i]=blurredDCT
        
        ####WRITE Baseline Images####
        imgBase = np.uint8(baselineDCT*255.0)
        print('Writing dct256_Base'+str(i)+'.png...')
        dirname=dirnames[0]
        imageio.imwrite(os.path.join(dirname,'dct256_Base'+str(i)+'.png'), imgBase)
        
        ###WRITE Baseline Coefficients###
        # Store data (serialize)
        with open(DEFAULT_BASELINE_DCT, 'wb') as fbase:
            pickle.dump(baseline_dict, fbase, protocol=pickle.HIGHEST_PROTOCOL)

        ####WRITE Blurred Images####
        imgBlur = np.uint8(blurredDCT*255.0)
        print('Writing dct256_Blurr'+str(i)+'.png...')
        dirname=dirnames[1]
        imageio.imwrite(os.path.join(dirname,'dct256_Blur'+str(i)+'.png'), imgBlur)
        
        ###WRITE Blurred Coefficients###
        # Store data (serialize)
        with open(DEFAULT_BLURRED_DCT, 'wb') as fblurred:
            pickle.dump(blurred_dict, fblurred, protocol=pickle.HIGHEST_PROTOCOL)
        
            
        
        
    ######WARNING START######
    #plt.show() #--> Figures created through the pyplot interface will consume too much memory until explicitly closed.
    ######WARNING END########    


