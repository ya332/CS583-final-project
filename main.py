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
    hash = "".join(map(str, img_hash.flatten()))
    return hash

def hammingDist(x, y):
    x = list(x)
    y = list(y)
    hd = sum([int(x[i]) ^ int(y[i]) for i in range(len(x))])
    return hd


def compareHash(query_hash, dict, r):
    retdict = []
    for hash in dict.keys():
        if hammingDist(query_hash, hash) <= r:
            retdict.append(hash)
    return retdict

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
        baseline_dict[baselineHash] = ['original_'+str(i+1)]
        blurred_dict[blurredHash] = ['blurred_'+str(i+1)]

        ####WRITE Baseline Images####
        # imgBase = np.uint8(baselineDCT*255.0)
        # print('Writing dct256_Base'+str(i)+'.png...')
        # dirname=dirnames[0]
        # imageio.imwrite(os.path.join(dirname,'dct256_Base'+str(i)+'.png'), imgBase)
        #
        # ###WRITE Baseline Coefficients###
        # # Store data (serialize)
        # with open(DEFAULT_BASELINE_DCT, 'wb') as fbase:
        #     pickle.dump(baseline_dict, fbase, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # ####WRITE Blurred Images####
        # imgBlur = np.uint8(blurredDCT*255.0)
        # print('Writing dct256_Blurr'+str(i)+'.png...')
        # dirname=dirnames[1]
        # imageio.imwrite(os.path.join(dirname,'dct256_Blur'+str(i)+'.png'), imgBlur)
        #
        # ###WRITE Blurred Coefficients###
        # # Store data (serialize)
        # with open(DEFAULT_BLURRED_DCT, 'wb') as fblurred:
        #     pickle.dump(blurred_dict, fblurred, protocol=pickle.HIGHEST_PROTOCOL)

    ### TESTING ON ALTERED DATASET ###
    # Loading the test datasets
    testdata_cropped = os.listdir('./cropped_img/')
    testdata_annotated = os.listdir('./annotated/')
    testdata_rot180 = os.listdir('./rot_180/')
    testdata_rot45 = os.listdir('./rot_45/')

    print('testdata_cropped')

    # testdata_cropped = readImages('./cropped_img/')
    # testdata_annotated = readImages('./annotated/')
    # testdata_rot180 = readImages('./rot_180/')
    # testdata_rot45 = readImages('./rot_45/')
    # cropped_dict, annotated_dict, rot180_dict, rot45_dict = {}, {}, {}, {}
    # print('Read test datasets')
    test_dirnames = ['crop_out', 'annotate_out', 'rot180_out', 'rot45_out']
    for dir in test_dirnames:
        if not os.path.exists(dir):
            os.mkdir(dir)
    print('Created output folders for test datasets')

    for i in range(len(testdata_cropped)):
        testimage_crop = imageio.imread('./cropped_img/'+testdata_cropped[i])[::,::].astype(np.float32)/255.
        testimg_crop_hash = computeHash(testimage_crop)
        crop_hash_baseline = compareHash(testimg_crop_hash, baseline_dict, 12)
        crop_hash_blurred = compareHash(testimg_crop_hash, blurred_dict, 12)
        # print('crop_hash_baseline', crop_hash_baseline, 'crop_hash_blurred', crop_hash_blurred)
        for h in crop_hash_baseline:
            baseline_dict[h].append(testdata_cropped[i])
        for h in crop_hash_blurred:
            blurred_dict[h].append(testdata_cropped[i])

        testimage_annotate = imageio.imread('./annotated/'+testdata_annotated[i])[::,::].astype(np.float32)/255.
        testimg_annotate_hash = computeHash(testimage_annotate)
        annotate_hash_baseline = compareHash(testimg_annotate_hash, baseline_dict, 10)
        annotate_hash_blurred = compareHash(testimg_annotate_hash, blurred_dict, 10)
        # print('annotate_hash_baseline', annotate_hash_baseline, 'annotate_hash_blurred', annotate_hash_blurred)
        for h in annotate_hash_baseline:
            baseline_dict[h].append(testdata_annotated[i])
        for h in annotate_hash_blurred:
            blurred_dict[h].append(testdata_annotated[i])

        testimage_rot180_im = imageio.imread('./rot_180/'+testdata_rot180[i])[::,::].astype(np.float32)/255.
        testimg_rot180_hash = computeHash(testimage_rot180_im)
        rot180_hash_baseline = compareHash(testimg_rot180_hash, baseline_dict, 15)
        rot180_hash_blurred = compareHash(testimg_rot180_hash, blurred_dict, 15)
        # print('rot180_hash_baseline', rot180_hash_baseline, 'rot180_hash_blurred', rot180_hash_blurred)
        for h in rot180_hash_baseline:
            baseline_dict[h].append(testdata_rot180[i])
        for h in rot180_hash_blurred:
            blurred_dict[h].append(testdata_rot180[i])

        testimage_rot45_im = imageio.imread('./rot_45/'+testdata_rot45[i])[::,::].astype(np.float32)/255.
        testimg_rot45_hash = computeHash(testimage_rot45_im)
        rot45_hash_baseline = compareHash(testimg_rot45_hash, baseline_dict, 17)
        rot45_hash_blurred = compareHash(testimg_rot45_hash, blurred_dict, 17)
        # print('rot45_hash_baseline', rot45_hash_baseline, 'rot45_hash_blurred', rot45_hash_blurred)
        for h in rot45_hash_baseline:
            baseline_dict[h].append(testdata_rot45[i])
        for h in rot45_hash_blurred:
            blurred_dict[h].append(testdata_rot45[i])

    print('baseline_dict')
    print("{:<8} {:<15}".format('Hash','Images'))
    for k, v in baseline_dict.items():
        print("{:<8} {:<100}".format(k, str(v)))
    print('blurred_dict')
    print("{:<8} {:<100}".format('Hash','Images'))
    for k, v in blurred_dict.items():
        print("{:<8} {:<100}".format(k, str(v)))


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
