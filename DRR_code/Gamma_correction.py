from __future__ import print_function
import numpy as np
import argparse
import cv2

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

path = r'D:/polynomial/resize_image/gamma_correction/real_one/normal_1.png'

original = cv2.imread(path)
k=1
for gamma in np.arange(0.0, 3.5, 0.5):
    if gamma == 1:
        continue
    gamma = gamma if gamma > 0 else 0.1
    adjusted = adjust_gamma(original, gamma=gamma)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow("Images", np.hstack([original, adjusted]))
    cv2.waitKey(0)
    cv2.imwrite('D:/polynomial/resize_image/gamma_correction/adjusted_one/adjusted_normal_(%d).png'%k,np.hstack([original, adjusted]))
    k=k+1







