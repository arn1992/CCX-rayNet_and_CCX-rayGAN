
from PIL import Image
import glob
import cv2
import numpy as np
import os
img_sum = 0
i=0
image_list = []
scale=1
count = 0
padColor=127
x='D:/research/Chest_CT_Dataset_JPG/Patient-/Study__CT/'
output = [dI for dI in os.listdir(x) if os.path.isdir(os.path.join(x,dI))]
#dataPath='D:/polynomial/CT/CHEST CT JPG/Patient-SHIM JEONG HWA/Study_862/Series_30298/'
#files_list = os.listdir(dataPath)
#files_list= files_list[0:128]
k=24

for filename in glob.glob('D:/research/Chest_CT_Dataset_JPG/Patient-/Study__CT/{}/*.jpg'.format(output[k])):

    if (count < 128):

    #for filename in files_list:
        print(filename)
        #im=Image.open(filename)

        #print(filename)
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        #print(scale)
        sh=int(img.shape[0]*scale)
        sw=int(img.shape[1]*scale)
        #print(h)
        #print(w)
        #print(sh)
        #print(sw)
        if (sh % 2==0) or (sw % 2==0) :
            sh=sh
            sw=sw
        else:
            sh=sh+1
            sw=sw+1
        p_l_t=int((h-sw)/2)
        p_r_b=int((w-sw)/2)
        #print(p_l_t,p_r_b)
        pad_left, pad_right, pad_top, pad_bot = p_l_t, p_r_b, p_l_t, p_r_b

        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                            borderType=cv2.BORDER_CONSTANT, value=padColor)
        #print(scaled_img.shape)
        #cv2.imshow('image', scaled_img)
        #cv2.waitKey(1000)
        img = scaled_img.astype('float64')
        img_sum += img
        scale = scale - 0.0005
        i=i+1
        count = count + 1


print(i)

img_sum = img_sum / i
#cv2.imwrite('D:/polynomial/CT/CHEST_CT_XRAY/Xray_Scaled_(%d).jpg'%k, img_sum)












'''

    img = cv2.imread(filename).astype('float64')
    img_sum += img
    image_list.append(im)
    i=i+1
print(i)
img_sum = img_sum / i
cv2.imwrite('D:/polynomial/CT/XRAY/new/Xray_1.jpg', img_sum)

#im.show()
    #image_list.append(im)
print(image_list)

'''

