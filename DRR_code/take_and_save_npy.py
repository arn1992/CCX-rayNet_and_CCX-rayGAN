
import torch.nn.functional as F
import torch
import os
import numpy as np
import torch.nn.functional as F
import torch
from skimage.transform import resize
'''
path = 'D:/research/3D_ARRAY_Reshape/'
x=[210, 340, 340]
npa = np.asarray(x, dtype=np.int32)
#print(npa)
j=1
for i in os.listdir(path):
    print(i)
    arr = np.load(os.path.join(path, i))
    #x, y, z = arr.shape
    #arr=arr.reshape(1,1,x,y,z)
    #print(arr.shape)
    #ct_array = torch.Tensor(arr)
    #print(ct_array.shape, ct_array.shape[0],ct_array.shape[1],ct_array.shape[2])
    #ct_array.shape[0]=166
    #ct_array.shape[1]=236
    #ct_array.shape[2]=236
    i = i[6:-4]
    #print(i)

    np.save('D:/polynomial/UNET2D/data/xray/test/normal_{}.npy'.format(i), arr)
    if (j==200):
        break
        j=j+1'''



path = 'D:/polynomial/UNET2D/data/xray/1/'
mask=[]
x=[128, 128, 128]
npa = np.asarray(x, dtype=np.int32)
for i in os.listdir(path):
    print(i)
    arr = np.load(os.path.join(path, i))
    image=F.interpolate(torch.Tensor(arr).unsqueeze(0).unsqueeze(0), npa.tolist(), mode='trilinear', align_corners=True).squeeze().detach().numpy()


    mask.append(image)

x_test=np.array(mask, dtype="float32")
print(x_test.shape)
max_value = float(x_test.max())
x_test[0] =x_test[0]/ 255
x_test[1] =x_test[1]/255
print(x_test.shape)

x_test = x_test.reshape((len(x_test), 128,128,128,1))
print(x_test.shape,type(x_test))