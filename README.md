# Perceptual Hasing as Adversarial Defense 
We investigate the use of Gaussian Blurring as a defense against black-box attacks on extended Yale B facial image dataset.

### Prerequisites
imageio==2.5.0  
numpy==1.16.2  
matplotlib==3.1.0  
opencv-python==4.1.0.25  
Python 3.x  

### How to Run
1)First clone the repo
```  
$ git clone https://github.com/ya332/CS583-final-project.git  
```

2)Then, run the main script entitled 'main.py'.  
```
$ python main.py  
```
(Based on symbolic linking, you might need to do python3 main.py)  
Output folders for test datasets will be created in your current working directory where you ran the main.py  

**Final Project Road Map / Requirements**

* [x] 1. Use Yale Face Database B dataset for our experiments(576 poses per 28 human subjects) **DONE**
* [x] 2. Calculate hash values for about 28 images(1 of each subject) in the dataset as a baseline, where we first
	* Calculate DCT frequency coefficients **DONE**
	* Get top 64 coefficients to calculate an 64 bit hash code( We anticipate 64 coefficients will be enough, we might need to experiment with this number, ie 128 or 32)(Note at this step we likely will need to resize/sample the images) **DONE**
* [x] 3. Save those 28 hash codes as baseline images. **DONE**
* [x] 4. Repeat the previous hash computation but make sure to blur the image with Gaussian Blurring before calculating the coefficients(sigma and kernel size need to be big enough so that image seems blurred to human eye) **DONE**
* [x] 5. Save those 28 hash codes as Gaussian Blurred ‘GB’ images. **DONE**
* [x] 6. Now create the malicious content/adversarial attacks by doing the following to all 28 images.
	* Cropping (so that hair and neck doesn’t show up and just face appears) **DONE**
	* Adding text on images (a simple text -unfortunately it is too early to determine the font size, font type or color, but we anticipate to add something simple such as a black text across the image saying ‘copyrighted’. Since we are aiming for robust face image detection, font size or font color shouldn’t matter on theory) **DONE**
	* Rotating images 180 degrees(so upside down) **DONE**
	* Rotating images 45 degrees(Crop the black pixels at the sides if the image is no longer rectangular) **DONE**
* [x] 7. Use the above altered images as the test dataset **DONE**
* [x] 8. Compute hash values for the test dataset. **DONE**
* [x] 9. Test if the test dataset images are detected as duplicates of the baseline images or as duplicates of the ‘GB’ images. **DONE**
* [x] 10. Verify if the hypothesis failed/succeeded. **DONE**
* [x] 11. Conclude with an explanation of the results, and determine whether future work is necessary. **DONE**
