import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
f1 = h5py.File('D:/polynomial/resize_image/CT/new/LIDC-IDRI-0003.20000101.3000611.1/ct_xray_data.h5', 'r')

list(f1.keys())
print(list(f1.keys()))
X1 = f1['ct']
print(np.amax(X1),np.amin(X1))
#y1=f1['y']
df1= np.array(X1.value)
print(df1.shape,type(df1))
#dfy1= np.array(y1.value)
print (df1.shape)
#print (dfy1.shape)'''

fig = plt.figure()
ax1 = fig.add_subplot(121)
# Bilinear interpolation - this will look blurry
ax1.imshow(X1, interpolation='bilinear', cmap=cm.Greys_r)

ax2 = fig.add_subplot(122)
# 'nearest' interpolation - faithful but blocky
ax2.imshow(X1, interpolation='nearest', cmap=cm.Greys_r)

plt.show()
'''
path='D:/polynomial/resize_image/plot'
for root, directories, files in os.walk(path):
    for name in files:
        filename=os.path.join(root, name)
        f1 = h5py.File(filename, 'r')
        print(list(f1.keys()))
        X1 = f1['xray1']
        df1 = np.array(X1.value)
        f1.close()'''









