import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(threshold=sys.maxsize)

a=np.load("D:/review/new_data_jsrt/img_crop_final_inverted.npy")
a = a.reshape((247, 256,256))
print(a[2])
b=np.zeros(shape = (247, 256, 256 ))
#print(a[1])
for i in range(247):
    inverted_img = (255.0 - a[i])
    #print(inverted_img)
    b[i]=inverted_img
print(b.shape,type(b))
print(b[2])
fig = plt.figure()
ax1 = fig.add_subplot(121)
# Bilinear interpolation - this will look blurry
ax1.imshow(b[1], interpolation='bilinear', cmap=cm.Greys_r)

ax2 = fig.add_subplot(122)
# 'nearest' interpolation - faithful but blocky
ax2.imshow(b[1], interpolation='nearest', cmap=cm.Greys_r)

plt.show()
b=b.reshape((247,256,256,1))


#print(b[1])


np.save("D:/review/new_data_jsrt/images/img_crop_final_inverted.npy", b)

