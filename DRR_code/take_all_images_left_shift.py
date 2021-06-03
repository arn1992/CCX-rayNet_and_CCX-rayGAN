
from PIL import Image
import glob
import cv2
import numpy as np
import os
img_sum = 0
i=0
count = 0
image_list = []
scale=1
padColor=127
for filename in glob.glob('D:/polynomial/CT/CHEST CT JPG/Patient-SHIM JEONG HWA/Study_862/2/*.jpg'): #assuming gif
    if (count<141):

        #im=Image.open(filename)
        #print(filename)
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        print(scale)
        sh=int(img.shape[0]*scale)
        sw=int(img.shape[1]*scale)
        print(h)
        print(w)
        print(sh)
        print(sw)
        if (sh % 2==0) or (sw % 2==0) :
            sh=sh
            sw=sw
        else:
            sh=sh+1
            sw=sw+1
        p_l_t=int((h-sw))
        p_r_b=int((w-sw))
        print(p_l_t,p_r_b)
        pad_left, pad_right, pad_top, pad_bot = p_l_t, 0, p_l_t, 0

        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3

        # scale and pad
        scaled_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                            borderType=cv2.BORDER_CONSTANT, value=padColor)
        print(scaled_img.shape)
        cv2.imshow('image', scaled_img)
        cv2.waitKey(1000)
        img = scaled_img.astype('float64')
        img_sum += img
        scale = scale - 0.002
        i=i+1
        count=count+1

print(i)

img_sum = img_sum / i
cv2.imwrite('D:/polynomial/CT/XRAY/Chest/Xray_scalled_Chest_left_shift_3.jpg', img_sum)












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

