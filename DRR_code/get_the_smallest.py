import os
import numpy as np
path = 'D:/research/3D_Array_with_resample/'
min_x=1000
min_y=1000
min_z=1000
j=''
k=''
l=''
for i in os.listdir(path):
    arr = np.load(os.path.join(path, i))
    x, y, z = arr.shape
    print(x,y,z,i)
    if x < min_x:
        j=i
        min_x = x
    if y < min_y:
        k=i
        min_y = y
    if z < min_z:
        l=i
        min_z = z
print(min_x,j)
print(min_y,k)
print(min_z,l)