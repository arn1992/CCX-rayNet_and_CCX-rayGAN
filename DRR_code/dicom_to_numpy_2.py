import pydicom
import os
import re
import numpy as np
#from matplotlib import pyplot, cm
import matplotlib.pyplot as plt
x='D:/research/done/'
output = [dI for dI in os.listdir(x) if os.path.isdir(os.path.join(x,dI))]
print(output)

num=1
for i in range(len(output)):

    PathDicom='D:/research/done/{}/'.format(output[i])
    temp = re.findall(r'\d+', output[i])
    res = list(map(int, temp))
    strings = [str(integer) for integer in res]
    a_string = "".join(strings)
    an_integer = int(a_string)

    print(an_integer)

    # print result

    #man=str(res)
    #print(man)

    print(PathDicom)
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])
    #print(lstFilesDCM[0],lstFilesDCM[1],lstFilesDCM[2])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)

        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
    print(ArrayDicom)
    np.save(os.path.join('3D_ARRAY', 'CT_3D_(%d).npy'%an_integer), ArrayDicom)
    num = num + 1

