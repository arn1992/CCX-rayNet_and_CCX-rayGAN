import os
import numpy as np
from skimage import io, exposure

def make_masks():
    path = 'D:/review/scr/scratch/fold1/masks/heart/'
    for i, filename in enumerate(os.listdir(path)):
        print(i,filename)
        left = io.imread('D:/review/scr/scratch/fold1/masks/left lung/' + filename[:-4] + '.gif')
        right = io.imread('D:/review/scr/scratch/fold1/masks/right lung/' + filename[:-4] + '.gif')
        heart = io.imread('D:/review/scr/scratch/fold1/masks/heart/' + filename[:-4] + '.gif')
        #lc=io.imread('D:/review/scr/scratch/fold2/masks/left clavicle/' + filename[:-4] + '.gif')
        #rc= io.imread('D:/review/scr/scratch/fold2/masks/right clavicle/' + filename[:-4] + '.gif')
        io.imsave('D:/review/scr/scratch/fold2/new/' + filename[:-4] + '.png', np.clip(left+right+heart, 0, 255))
        print ('Mask', i, filename)


make_masks()