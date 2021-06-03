import cv2
import numpy as np

img1 = cv2.imread('abc.jpg')
a = np.double(img1)
b = a + 0.5
img2 = np.uint8(b)
cv2.imshow("frame",img1)
cv2.imshow("frame2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()