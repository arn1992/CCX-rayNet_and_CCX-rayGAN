import numpy as np
import nibabel as nib

img = nib.load("D:/research/AAAI/LIDC-IDRI-0026.20000101.3000519.1_gen.nii.gz")

a = np.array(img.dataobj)
print(a.shape)