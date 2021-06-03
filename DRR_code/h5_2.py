# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:26:13 2020

@author: suppo
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
'''f1 = h5py.File('D:/polynomial/resize_image/CT/ct_xray_data.h5', 'r')

list(f1.keys())
print(list(f1.keys()))
X1 = f1['foo']
#y1=f1['y']
df1= np.array(X1.value)
print(df1,type(df1))
#dfy1= np.array(y1.value)
print (df1.shape)
#print (dfy1.shape)'''


im=[]
path='D:/polynomial/resize_image/plot'
for root, directories, files in os.walk(path):
    for name in files:
        filename=os.path.join(root, name)
        f1 = h5py.File(filename, 'r')
        X1 = f1['xray1']
        df1 = np.array(X1.value)
        x_img = np.resize(df1, (256, 256,1))
        #print(x_img.shape)
        # print(df1, type(df1))
        im.append(x_img)
        #print(type(im))

X = np.array(im, dtype="float32")
X = X.reshape((1018, 256, 256,1))
print(X.shape,type(X))
#np.save('D:/aminur/data/X2CT.npy',X)'''


'''
f1 = h5py.File('D:/polynomial/resize_image/CT/ct_xray_data.h5', 'r')
X1=f1['xray1']
df1 = np.array(X1.value)
print(type(X1))
print(list(f1.keys()))
f1.close()'''
'''hf = h5py.File('D:/polynomial/resize_image/CT/ct_xray_data.h5', 'w')



hf['foo'] = np.random.random(100)
hf['bar'] = np.random.random(100) + 10.
hf['xray1']=df1
hf.close()'''



