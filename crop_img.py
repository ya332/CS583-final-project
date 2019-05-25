import os
from PIL import Image

dirname = 'cropped_img'

if not os.path.exists(dirname):
    os.mkdir(dirname)

original_img = os.listdir('./img')
print(original_img)

i = 1

for img in original_img:
    im = Image.open('./img/'+img+'/image0.jpg')
    # Size of cropped img: 148 x 157
    img_crop = im.crop((10, 15, 158, 172)) # co-ordinates of tuple -> (x1, y1, x2, y2)
    img_crop.save(os.path.join(dirname, 'cropped_img'+str(i)+'.png'))
    i += 1
print('Cropped images saved in cropped_img directory')
