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
    if imgFolder == 'img/':
        images = [imageio.imread('img/'+fn+'/image0.jpg')[::,::].astype(np.float32)/255. for fn in filenames]#glob.glob(imgFolder+'*.jpg')]
    else:
        images = [imageio.imread(imgFolder+fn)[::,::].astype(np.float32)/255. for fn in filenames]
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

def computeHash(img):
    img_resize = cv2.resize(img, (32, 32))
    img_DCT = cv2.dct(img_resize)
    low_freq_dct = img_DCT[:8, 1:9]
    avg = np.mean(low_freq_dct)
    img_hash = np.where(low_freq_dct > avg, 1, 0)
    hash = int("".join(map(str, img_hash.flatten())))
    return hash

def compareHash(input_hash, dict):
    pass
    # currently working on what technique to use for comparing hashes

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
        baselineHash, blurredHash= computeHash(images[i]), computeHash(blurredImage)
        baseline_dict[baselineHash] = ['original_'+i+1]
        blurred_dict[blurredHash] = ['blurred_'+i+1]

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

    ### TESTING ON ALTERED DATASET ###
    # Loading the test datasets
    testdata_cropped = readImages('./cropped_img/')
    testdata_annotated = readImages('./annotated/')
    testdata_rot180 = readImages('./rot_180/')
    testdata_rot45 = readImages('./rot_45/')
    cropped_dict, annotated_dict, rot180_dict, rot45_dict = {}, {}, {}, {}
    print('Read test datasets')
    test_dirnames = ['crop_out', 'annotate_out', 'rot180_out', 'rot45_out']
    for dir in test_dirnames:
        if not os.path.exists(dir):
            os.mkdir(dir)
    print('Created output folders for test datasets')

    for i in range(len(testdata_cropped)):
        crop_hash = computeHash(testdata_cropped[i])

        annotate_hash = computeHash(testdata_annotated[i])

        rot180_hash = computeHash(testdata_rot180[i])

        rot45_hash = computeHash(testdata_rot45[i])


        # ###WRITE Cropped Image DCT###
        # imgCrop = np.uint8(crop_DCT*255.0)
        # imageio.imwrite(os.path.join(test_dirnames[0],'dct256_Crop'+str(i)+'.png'), imgCrop)
        #
        # ###WRITE Annotated Image DCT###
        # imgAnnotate = np.uint8(annotate_DCT*255.0)
        # imageio.imwrite(os.path.join(test_dirnames[1],'dct256_Annotate'+str(i)+'.png'), imgAnnotate)
        #
        # ###WRITE Rotated 180 Image DCT###
        # imgRot180 = np.uint8(rot180_DCT*255.0)
        # imageio.imwrite(os.path.join(test_dirnames[2],'dct256_Rot180'+str(i)+'.png'), imgRot180)
        #
        # ###WRITE Rotated 45 Image DCT###
        # imgRot45 = np.uint8(rot45_DCT*255.0)
        # imageio.imwrite(os.path.join(test_dirnames[3],'dct256_Rot45'+str(i)+'.png'), imgRot45)


    ######WARNING START######
    #plt.show() #--> Figures created through the pyplot interface will consume too much memory until explicitly closed.
    ######WARNING END########
