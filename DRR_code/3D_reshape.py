import torch.nn.functional as F
import torch
import os
import numpy as np
path = 'D:/research/3D_Array_with_resample/'
x=[340, 340, 340]
npa = np.asarray(x, dtype=np.int32)
#print(npa)
a=0
b=0
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
    #i = i[:-4]
    #print(i)

    image=F.interpolate(torch.Tensor(arr).unsqueeze(0).unsqueeze(0), npa.tolist(), mode='trilinear', align_corners=True).squeeze().detach().numpy()
    #print(image.shape,type(image))
    if image.max()>a:
        a=image.max()
    if image.min()<b:
        b=image.min()
    #np.save(os.path.join('exam1', '{}.npy'.format(i)), image)
print(a,b)