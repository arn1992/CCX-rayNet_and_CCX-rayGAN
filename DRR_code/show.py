import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

np.set_printoptions(threshold=sys.maxsize)

a=np.load("D:/review/new_data_jsrt/result_of_segmentation/map.npy")
print(a.shape)
a = a.reshape((1018, 256,256))
#print(a[1])
#print(np.ptp(x,axis=1))

print(a.min(),a.max())

f1 = h5py.File('D:/polynomial/resize_image/CT/new/LIDC-IDRI-0015.20000101.3000610.1/ct_xray_data.h5', 'r')

list(f1.keys())
print(list(f1.keys()))
X1 = f1['xray2']
#y1=f1['y']
df1= np.array(X1.value)
print(df1.shape,type(df1))
cv2.imwrite('D:/research/AAAI/good result from segmentation map/15_xray_2.png', df1)
#dfy1= np.array(y1.value)
print (df1.shape)
y1 = f1['map']
#y1=f1['y']
df2= np.array(y1.value)
#cv2.imwrite('D:/research/AAAI/good result from segmentation map/6_xray_map.png', df2)
fig = plt.figure()
ax1 = fig.add_subplot(121)
# Bilinear interpolation - this will look blurry
ax1.imshow(a[200], interpolation='bilinear', cmap=cm.Greys_r)

ax2 = fig.add_subplot(122)
# 'nearest' interpolation - faithful but blocky
ax2.imshow(y1, interpolation='nearest', cmap=cm.Greys_r)

#plt.show()

