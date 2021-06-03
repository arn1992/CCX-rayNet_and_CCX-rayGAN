import os
import re
import numpy as np
import cv2
from PIL import Image

#from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
x='D:/polynomial/resize_image/RESULT_x/'
output = [dI for dI in os.listdir(x) if os.path.isdir(os.path.join(x,dI))]
print(output)
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

for i in range(len(output)):
    Path='D:/polynomial/resize_image/RESULT_x/{}/'.format(output[i])
    '''
    temp = re.findall(r'\d+', output[i])
    res = list(map(int, temp))
    strings = [str(integer) for integer in res]
    a_string = "".join(strings)
    an_integer = int(a_string)
    print(an_integer)
    '''
    h=output[i] #dest. folder name
    #print(h)
    if os.path.exists("D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/{}".format(h)) is False:
        os.mkdir("D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/{}".format(h))


    lstFilesDCM = []  # create an empty list
    imagename=[]
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".png" in filename.lower():  # check whether the file's DICOM
                imagename.append(filename)
                lstFilesDCM.append(os.path.join(dirName, filename))
    #k=lstFilesDCM
    # loop through all the png files
    i=0
    for filepath in lstFilesDCM:
        print(filepath)
        p=imagename[i]
        p=p[:-4]
        #print(p)

        original = cv2.imread(filepath)
        d = "data{}".format(h)

        adjusted = adjust_gamma(original, gamma=2.5)

        # cv2.imshow("Images", adjusted)
        # cv2.waitKey(0)
        cv2.imwrite('D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/{}/gamma_{}.png'.format(h, p),adjusted)
        colorImage = Image.open('D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/{}/gamma_{}.png'.format(h, p))
        transposed = colorImage.transpose(Image.ROTATE_180)
        transposed.save('D:/research/RESULT_CHEST_CT_gamma_correction/RESULT_x/{}/gamma_{}.png'.format(h, p))

        i=i+1






