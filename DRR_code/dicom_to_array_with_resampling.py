import SimpleITK as sitk
import numpy as np
import os
import pydicom
import torch.nn.functional as F
import torch

def load_scan(path):
    slices = [pydicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness


    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    return slices, image
def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape).astype('int32')
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print(new_shape, type(new_shape))
    f=torch.Tensor(image).unsqueeze(0).unsqueeze(0)
    print(f.shape)

    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    image = F.interpolate(torch.Tensor(image).unsqueeze(0).unsqueeze(0), new_shape.tolist(), mode='trilinear',
                             align_corners=True).squeeze().detach().numpy()

    return image, new_spacing



def get_pixels_hu(image, slice_list):
    image[image == -2000] = 0


    for slice_number in range(len(slice_list)):

        intercept = slice_list[slice_number].RescaleIntercept
        slope = slice_list[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

INPUT_FOLDER = 'D:/research/done/'
patients = os.listdir(INPUT_FOLDER)
patients.sort()
pat=[]
for patient in patients:
    print(patient)
    slice_list, image = load_scan(os.path.join(INPUT_FOLDER, patient))
    #print(type(slice_list))
    pat=slice_list
    #print("slice thickness: ",slice_list[0].SliceThickness)
    #print("Pixel Spacing (row, col): (%f, %f) " % (slice_list[0].PixelSpacing[0], slice_list[0].PixelSpacing[1]))
    image = get_pixels_hu(image, slice_list)
    print(image.shape)
    #print(image.shape,type(image))
    image_res, spacing = resample(image, pat, [1, 1, 1])
    print(image_res.shape)
    #np.save(os.path.join('3D_Array_with_resample', 'CT_3D_{}.npy'.format(patient)), image_res)
    #out=sitk.GetImageFromArray(image_res)
    #sitk.WriteImage(out, os.path.join('3D_Array_with_resample_nii_gz', 'CT_3D_{}.nii.gz'.format(patient)))