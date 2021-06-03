import numpy as np
import nibabel as nib
import os
path='D:/research/90_epoch/'

for i in os.listdir(path):
    print(i)

    img = nib.load(os.path.join(path, i))

    a = np.array(img.dataobj)
    i = i[:-8]
    print(a)
    np.save(os.path.join(path, '{}.npy'.format(i)), a)