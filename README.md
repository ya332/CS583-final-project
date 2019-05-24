# CS583-final-project
CS 583 Final Project Repo for Collaboration


1. Use Yale Face Database B dataset for our experiments(576 poses per 28 human subjects) **DONE**
2. Calculate hash values for about 28 images(1 of each subject) in the dataset as a baseline, where we first
	* Calculate DCT frequency coefficients **DONE**
	* Get top 64 coefficients to calculate an 64 bit hash code( We anticipate 64 coefficients will be enough, we might need to experiment with this number, ie 128 or 32)(Note at this step we likely will need to resize/sample the images) **DONE**
3. Save those 28 hash codes as baseline images. **DONE**
4. Repeat the previous hash computation but make sure to blur the image with Gaussian Blurring before calculating the coefficients(sigma and kernel size need to be big enough so that image seems blurred to human eye) **DONE**
5. Save those 28 hash codes as Gaussian Blurred ‘GB’ images. **DONE**
6. Now create the malicious content/adversarial attacks by doing the following to all 28 images.
	* Cropping (so that hair and neck doesn’t show up and just face appears) **IN PROGRESS by Manika**
	* Adding text on images (a simple text -unfortunately it is too early to determine the font size, font type or color, but we anticipate to add something simple such as a black text across the image saying ‘copyrighted’. Since we are aiming for robust face image detection, font size or font color shouldn’t matter on theory) **DONE**
	* Rotating images 180 degrees(so upside down) **DONE**
	* Rotating images 45 degrees(Crop the black pixels at the sides if the image is no longer rectangular) **IN PROGRESS by Yigit**
7. Use the above altered images as the test dataset
8. Compute hash values for the test dataset.
9. Test if the test dataset images are detected as duplicates of the baseline images or as duplicates of the ‘GB’ images.
10. Verify if the hypothesis failed/succeeded.
11. Conclude with an explanation of the results, and determine whether future work is necessary.

