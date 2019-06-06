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
    """blur the image with Gaussian Smoothing technique"""
    #kernel = cv2.getGaussianKernel(ksize,sigma)
    dst = np.zeros_like(img)
    cv2.GaussianBlur(src=img,dst=dst,ksize=ksize,sigmaX=0)
    return dst

def plotFace(original,blurred):
    """Helper function to display an side-by-side comparison between the original and the blurred"""
    plt.subplot(121),plt.imshow(original,cmap=cm.Greys_r),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blurred,cmap=cm.Greys_r),plt.title('Gaussian Blurred')
    plt.xticks([]), plt.yticks([])
    return None


def computePerceptualHash(img, length=64):
    """Compute the hash based on Discrete Cosine Transformatio of an image"""
    img_resize = cv2.resize(img, (32, 32))
    img_DCT = cv2.dct(img_resize)
    
    if length==64:
        low_freq_dct = img_DCT[1:9, 1:9]
    elif length==32:
        low_freq_dct = img_DCT[1:7, 1:7]    
        
    avg = np.mean(low_freq_dct)
    img_hash = np.where(low_freq_dct > avg, 1, 0)
    hash = "".join(map(str, img_hash.flatten()[0:length]))
    return hash

"""
def computePerceptualHash_32bit(img):
    #DEPRECATED
    img_resize = cv2.resize(img, (32, 32))
    img_DCT = cv2.dct(img_resize)
    low_freq_dct = img_DCT[1:7, 1:7]
    avg = np.mean(low_freq_dct)
    img_hash = np.where(low_freq_dct > avg, 1, 0)
    hash = "".join(map(str, img_hash.flatten()[0:32]))
    return hash

def computeAverageHash_32bit(img):
    #DEPRECATED
    img_resize = cv2.resize(img, (6, 6))
    avg = np.mean(img_resize)
    img_hash = np.where(img_resize > avg, 1, 0)
    hash = "".join(map(str, img_hash.flatten()[0:32]))
    return hash
"""

def computeAverageHash(img, length = 64):
    """Compute the hash of an image based on Average Hash Method"""
    if length == 64:
        img_resize = cv2.resize(img, (8, 8))
    elif length == 32:
        img_resize = cv2.resize(img, (6, 6))
    avg = np.mean(img_resize)
    img_hash = np.where(img_resize > avg, 1, 0)
    hash = "".join(map(str, img_hash.flatten()[0:length]))
    return hash

def hammingDist(x, y):
    """Calculate the Humming Distance between two hashes"""
    hd = 0
    for ch1, ch2 in zip(x, y):
        if ch1 != ch2:
            hd += 1
    return hd


def compareHash(query_hash, dict, r):
    """Compare two hashes based on the input threshold value"""
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

#Simulation Code
if __name__=="__main__":
    images = readImages('img/')
    print('Found images:',len(images))
    baseline_dict,blurred_dict={},{}

    dirnames = ['base_out','blurred_out']
    for d in dirnames:
        if d not in os.listdir(os.getcwd()):
            os.mkdir(d)
            print('Creating output folder:',d)

    for i in range(len(images)):
        plt.figure(i)
        blurredImage = gaussianBlur(images[i])
        #plotFace(images[i],blurredImage)
        baselineHash, blurredHash= computePerceptualHash(images[i],length=32), computePerceptualHash(blurredImage, length=32)
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

    #Initialize the y axis variables for accuracy plotting
    y_base_ann, y_base_crop, y_base_rot180 ,y_base_rot45 = [], [], [], []
    y_blur_ann, y_blur_crop, y_blur_rot180 ,y_blur_rot45 = [], [], [], []

    #Change the length here for different hash bit lengths
    length=32
    for th in range(1, length+1):
        for i in range(len(testdata_cropped)):
            
            testimage_crop = imageio.imread('./cropped_img/'+testdata_cropped[i])[::,::].astype(np.float32)/255.
            testimg_crop_hash = computePerceptualHash(testimage_crop, length)
            crop_hash_baseline = compareHash(testimg_crop_hash, baseline_dict, th) 
            crop_hash_blurred = compareHash(testimg_crop_hash, blurred_dict, th)
            for h in crop_hash_baseline:
                baseline_dict[h].append(testdata_cropped[i])
            for h in crop_hash_blurred:
                blurred_dict[h].append(testdata_cropped[i])

            testimage_annotate = imageio.imread('./annotated/'+testdata_annotated[i])[::,::].astype(np.float32)/255.
            testimg_annotate_hash = computePerceptualHash(testimage_annotate, length)
            annotate_hash_baseline = compareHash(testimg_annotate_hash, baseline_dict, th) 
            annotate_hash_blurred = compareHash(testimg_annotate_hash, blurred_dict, th)
            for h in annotate_hash_baseline:
                baseline_dict[h].append(testdata_annotated[i])
            for h in annotate_hash_blurred:
                blurred_dict[h].append(testdata_annotated[i])

            testimage_rot180_im = imageio.imread('./rot_180/'+testdata_rot180[i])[::,::].astype(np.float32)/255.
            testimg_rot180_hash = computePerceptualHash(testimage_rot180_im, length)
            rot180_hash_baseline = compareHash(testimg_rot180_hash, baseline_dict, th) 
            rot180_hash_blurred = compareHash(testimg_rot180_hash, blurred_dict, th)
            for h in rot180_hash_baseline:
                baseline_dict[h].append(testdata_rot180[i])
            for h in rot180_hash_blurred:
                blurred_dict[h].append(testdata_rot180[i])
            
            testimage_rot45_im = imageio.imread('./rot_45/'+testdata_rot45[i])[::,::].astype(np.float32)/255.
            testimg_rot45_hash = computePerceptualHash(testimage_rot45_im , length)
            rot45_hash_baseline = compareHash(testimg_rot45_hash, baseline_dict, th) 
            rot45_hash_blurred = compareHash(testimg_rot45_hash, blurred_dict, th)
            for h in rot45_hash_baseline:
                baseline_dict[h].append(testdata_rot45[i])
            for h in rot45_hash_blurred:
                blurred_dict[h].append(testdata_rot45[i])

        #Calculate baseline accuracies
        final_baseline, final_blurred = {}, {}
        acc_annotate, acc_crop, acc_rot180, acc_rot45 = 0, 0, 0, 0
      
        i=1
        for k in baseline_dict.keys():
            final_baseline[baseline_dict[k][0]] = baseline_dict[k][1:]
            
            i=str(i)
            if ('image_annotated_'+i+'.png' in baseline_dict[k][1:]):
                acc_annotate += 1

            elif ('cropped_img'+i+'.png' in baseline_dict[k][1:]):
                acc_crop += 1
            elif ('image_45_'+i+'.png' in baseline_dict[k][1:]):
                acc_rot45 += 1
            elif ('image_180_'+i+'.png' in baseline_dict[k][1:]):
                acc_rot180 += 1
            i=int(i)
            i+=1
        print('Threshold:', th)
        print('Baseline final accuracies:',acc_annotate, acc_crop, acc_rot180, acc_rot45)
        y_base_ann.append(acc_annotate)
        y_base_crop.append(acc_crop)
        y_base_rot180.append(acc_rot180)
        y_base_rot45.append(acc_rot45)
        
        gb_acc_annotate, gb_acc_crop, gb_acc_rot180, gb_acc_rot45 = 0, 0, 0, 0

        #Calculate blurred accuracies
        i=1
        for k in blurred_dict.keys():
            final_blurred[blurred_dict[k][0]] = blurred_dict[k][1:]
            i=str(i)
            if ('image_annotated_'+i+'.png' in blurred_dict[k][1:]):
                gb_acc_annotate += 1
            elif ('cropped_img'+i+'.png' in blurred_dict[k][1:]):
                gb_acc_crop += 1
            elif ('image_45_'+i+'.png' in blurred_dict[k][1:]):
                gb_acc_rot45 += 1
            elif ('image_180_'+i+'.png' in blurred_dict[k][1:]):
                gb_acc_rot180 += 1
            i=int(i)
            i+=1
        print('Blurred final accuracies:', gb_acc_annotate, gb_acc_crop, gb_acc_rot180, gb_acc_rot45)
        y_blur_ann.append(gb_acc_annotate)
        y_blur_crop.append(gb_acc_crop)
        y_blur_rot180.append(gb_acc_rot180)
        y_blur_rot45.append(gb_acc_rot45)

    #Plot the results
    y_base_ann = [x/28*100 for x in y_base_ann]
    y_base_crop = [x/28*100 for x in y_base_crop]
    y_base_rot180 = [x/28*100 for x in y_base_rot180]
    y_base_rot45 = [x/28*100 for x in y_base_rot45]

    y_blur_ann = [x/28*100 for x in y_blur_ann]
    y_blur_crop = [x/28*100 for x in y_blur_crop]
    y_blur_rot180 = [x/28*100 for x in y_blur_rot180]
    y_blur_rot45 = [x/28*100 for x in y_blur_rot45]

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots/')

    plt.figure(1)
    plt.plot(range(1,length+1),y_base_ann)
    plt.title('Accuracy vs Threshold for Annotated Baseline Images for DCT'+str(length)+'-Bits')
    my_file='fig25.png'
    plt.savefig(results_dir + my_file)

    plt.figure(2)
    plt.plot(range(1,length+1),y_base_crop)
    plt.title('Accuracy vs Threshold for Cropped Baseline Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig26.png'
    plt.savefig(results_dir + my_file)  

    plt.figure(3)
    plt.plot(range(1,length+1),y_base_rot180)
    plt.title('Accuracy vs Threshold for 180 Degrees Rotated Baseline Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig27.png'
    plt.savefig(results_dir + my_file)

    plt.figure(4)
    plt.plot(range(1,length+1),y_base_rot45)
    plt.title('Accuracy vs Threshold for 45 Degrees Rotated Baseline Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig28.png'
    plt.savefig(results_dir + my_file)

    plt.figure(5)
    plt.plot(range(1,length+1),y_blur_ann)
    plt.title('Accuracy vs Threshold for Annotated Blurred Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig29.png'
    plt.savefig(results_dir + my_file)
    
    plt.figure(6)
    plt.plot(range(1,length+1),y_blur_crop)
    plt.title('Accuracy vs Threshold for Cropped Blurred Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig30.png'
    plt.savefig(results_dir + my_file)
    
    plt.figure(7)
    plt.plot(range(1,length+1),y_blur_rot180)
    plt.title('Accuracy vs Threshold for 180 Degrees Rotated Blurred Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig31.png'
    plt.savefig(results_dir + my_file)
    
    plt.figure(8)
    plt.plot(range(1,length+1),y_blur_rot45)
    plt.title('Accuracy vs Threshold for 45 Degrees Rotated Blurred Images for DCT Hash'+str(length)+'-Bits')
    my_file='fig32.png'
    plt.savefig(results_dir + my_file)
    
    #plt.show()
    #plt.close()
    
'''
        print('baseline_dict. threshold =', th)
        print("{:<8} {:<100}".format('Hash','Images'))
        for k, v in final_baseline.items():
            print("{:<8} {:<100}".format(k, str(v)))
        print('blurred_dict')
        print("{:<8} {:<100}".format('Hash','Images'))
        for k, v in final_blurred.items():
            print("{:<8} {:<100}".format(k, str(v)))
'''


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
    #plt.show() #--> Figures created through the pyplot interface will consume too much memory until explicitly closed because of in-memory RAM usage
    ######WARNING END########
