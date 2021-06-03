import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(threshold=sys.maxsize)

a=np.load("D:/review/new_data_jsrt/img_crop_final_inverted.npy")
a = a.reshape((247, 256,256))
b=np.load("D:/review/new_data_jsrt/jsrt_mask_heart.npy")

b = b.reshape((247, 256,256))
#print(a[1])
#inverted_img = (255.0 - a[1])
#final = inverted_img / 255.0
#print(np.ptp(x,axis=1))
#print("inverted: ",inverted_img)
print(a.min(),a.max())

f1 = h5py.File('D:/polynomial/resize_image/CT/ct_xray_data.h5', 'r')

list(f1.keys())
print(list(f1.keys()))
X1 = f1['xray1']
#y1=f1['y']
df1= np.array(X1.value)
print(df1.shape,type(df1))
#dfy1= np.array(y1.value)
print (df1.shape)

fig = plt.figure()
ax1 = fig.add_subplot(121)
# Bilinear interpolation - this will look blurry
ax1.imshow(a[3], interpolation='bilinear', cmap=cm.Greys_r)

ax2 = fig.add_subplot(122)
# 'nearest' interpolation - faithful but blocky
ax2.imshow(b[3], interpolation='nearest', cmap=cm.Greys_r)

plt.show()

