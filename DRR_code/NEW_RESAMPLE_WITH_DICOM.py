from PIL import Image
import glob
import cv2
import numpy as np
import os
import pydicom
# import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.misc import imsave
import torch.nn.functional as F
import torch
import multiprocessing

def normal(patient_folder, result_path):
    img_sum = 0
    i=0

    for filename in glob.glob(patient_folder+'/*.png'):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype('float64')
        img_sum += img
        i += 1

    img_sum = img_sum / i
    cv2.imwrite(result_path, img_sum)

def CT2Xray_center_scale(patient_folder, result_path):
    img_sum = 0
    i=0
    image_list = []
    scale=1
    padColor=127
    for filename in glob.glob(patient_folder+'/*.png'): #assuming gif
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape[:2]

        sh=int(h*scale)
        sw=int(w*scale)
        print(h)
        print(w)
        print(sh)
        print(sw)
        '''
        if (sh % 2==0) or (sw % 2==0) :
            sh=sh
            sw=sw
        else:
            sh=sh+1
            sw=sw+1
        '''
        p_l_t=int((h-sh))
        p_r_b=int((w-sw))
        print(p_l_t, p_r_b, 'padding')
        if (p_r_b>=0) or (p_l_t>=0):
            pad_left, pad_right, pad_top, pad_bot = p_l_t, p_r_b, p_l_t, p_r_b# Scale coordinate

            if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
                padColor = [padColor]*3

            # scale and pad
            scaled_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                                borderType=cv2.BORDER_CONSTANT, value=padColor)
            print(scaled_img.shape)
            dw, dh = scaled_img.shape[:2]
            if (h < dh) or (w < dw) or (h > dh) or (w > dw):
                scaled_img = cv2.resize(scaled_img, (w, h), interpolation=cv2.INTER_AREA)
            #cv2.imshow('image', scaled_img)
            #cv2.waitKey(1000)
            img = scaled_img.astype('float64')
            img_sum += img
            scale = scale - 0.00005
        i=i+1

    print(i)

    img_sum = img_sum / i
    cv2.imwrite(result_path, img_sum)

def CT2Xray_right_shift(patient_folder, result_path):
    img_sum = 0
    i=0
    image_list = []
    scale=1
    padColor=127
    for filename in glob.glob(patient_folder+'/*.png'): #assuming gif
        #print(filename)
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        sh=int(h*scale)
        sw=int(w*scale)
        print(h)
        print(w)
        print(sh)
        print(sw)
        '''
        if (sh % 2==0) or (sw % 2==0) :
            sh=sh
            sw=sw
        else:
            sh=sh+1
            sw=sw+1
            '''
        p_l_t=int((h-sh))
        p_r_b=int((w-sw))
        print(p_l_t, p_r_b, 'padding')
        if (p_r_b>=0) or (p_l_t>=0):
            pad_left, pad_right, pad_top, pad_bot = 0, p_r_b, 0, p_r_b# Scale coordinate

            if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
                padColor = [padColor]*3

            # scale and pad
            scaled_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                                borderType=cv2.BORDER_CONSTANT, value=padColor)
            print(scaled_img.shape)
            dw, dh = scaled_img.shape[:2]
            if (h < dh) or (w < dw) or (h > dh) or (w > dw):
                scaled_img = cv2.resize(scaled_img, (w, h), interpolation=cv2.INTER_AREA)
            #cv2.imshow('image', scaled_img)
            #cv2.waitKey(1000)
            img = scaled_img.astype('float64')
            img_sum += img
            scale = scale - 0.00005
        i=i+1

    print(i)

    img_sum = img_sum / i
    cv2.imwrite(result_path, img_sum)

def CT2Xray_left_shift(patient_folder, result_path):
    img_sum = 0
    i=0
    image_list = []
    scale=1
    padColor=127
    for filename in glob.glob(patient_folder+'/*.png'): #assuming gif
        #print(filename)
        img = cv2.imread(filename)
        h, w = img.shape[:2]

        sh=int(h*scale)
        sw=int(w*scale)
        print(h)
        print(w)
        print(sh)
        print(sw)
        '''
        if (sh % 2==0) or (sw % 2==0) :
            sh=sh
            sw=sw
        else:
            sh=sh+1
            sw=sw+1
            '''

        p_l_t=int((h-sh))
        p_r_b=int((w-sw))
        print(p_l_t, p_r_b, 'padding')
        if (p_r_b>=0) or (p_l_t>=0):
            pad_left, pad_right, pad_top, pad_bot = p_l_t, 0, p_l_t, 0# Scale coordinate

            if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
                padColor = [padColor]*3

            # scale and pad
            scaled_img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                                borderType=cv2.BORDER_CONSTANT, value=padColor)
            print(scaled_img.shape)
            dw, dh = scaled_img.shape[:2]
            if (h < dh) or (w < dw) or (h > dh) or (w > dw):
                scaled_img = cv2.resize(scaled_img, (w, h), interpolation=cv2.INTER_AREA)
            #cv2.imshow('image', scaled_img)
            #cv2.waitKey(1000)
            img = scaled_img.astype('float64')
            img_sum += img
            scale = scale - 0.00005
        i=i+1

    print(i)

    img_sum = img_sum / i
    cv2.imwrite(result_path, img_sum)


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



def resample_spacing(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
    print('z, x, y spacing:', spacing)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape).astype('int32')
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # image = scipy.ndimage.zoom(image, real_resize_factor, order=3) # Problem with scipy, change interpolation into pytorch

    image = F.interpolate(torch.Tensor(image).unsqueeze(0).unsqueeze(0), new_shape.tolist(), mode='trilinear',
                             align_corners=True).squeeze().detach().numpy()
    return image, new_spacing

def resample_voxel(image, scan, new_spacing=[128, 128, 128]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
    print('z, x, y spacing:', spacing)

    new_shape = [128, 128, 128]
    image = F.interpolate(torch.Tensor(image).unsqueeze(0).unsqueeze(0), new_shape, mode='trilinear',
                             align_corners=True).squeeze().detach().numpy()
    print(image.shape)
    return image, new_spacing




if __name__ == '__main__':
    # Load the dicom folder
    save_axis = 'y' # orientation of the slices
    need_resample = True # Need to do resample or not
    INPUT_FOLDER = './CT'
    RESAMPLE_FOLDER_x = './RESAMPLE_x'
    RESAMPLE_FOLDER_y = './RESAMPLE_y'
    RESAMPLE_FOLDER_z = './RESAMPLE_z'

    RESULT_FOLDER_x = './RESULT_x'
    RESULT_FOLDER_y = './RESULT_y'
    RESULT_FOLDER_z = './RESULT_z'

    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    for patient in patients:
        print(patient)
        slice_list, image = load_scan(os.path.join(INPUT_FOLDER, patient))
        image = get_pixels_hu(image, slice_list)

        # Get HU value
        plt.hist(image.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        #plt.show()

        # Plot the intermediate slice
        plt.imshow(image[60], cmap=plt.cm.gray)
        #plt.show()

        # Resample to 1mm, 1mm, 1mm
        pix_resampled, spacing = resample_spacing(image, slice_list, [1, 1, 1])
        print("Shape before resampling\t", image.shape)
        print("Shape after resampling\t", pix_resampled.shape)
        # print(np.min(pix_resampled), np.max(pix_resampled))
        pix_resampled[pix_resampled>304] = 304
        pix_resampled[pix_resampled<-79] = -79
        # print(np.min(pix_resampled), np.max(pix_resampled))

        if not os.path.exists(os.path.join(RESAMPLE_FOLDER_x, patient)): os.mkdir(os.path.join(RESAMPLE_FOLDER_x, patient))
        if not os.path.exists(os.path.join(RESAMPLE_FOLDER_y, patient)): os.mkdir(os.path.join(RESAMPLE_FOLDER_y, patient))
        if not os.path.exists(os.path.join(RESAMPLE_FOLDER_z, patient)): os.mkdir(os.path.join(RESAMPLE_FOLDER_z, patient))

        # Save the slices
        # if save_axis == 'z':
        for i in range(pix_resampled.shape[0]):
            imsave(os.path.join(RESAMPLE_FOLDER_z, patient, str(i) + '.png'), pix_resampled[i, :, :])

        # if save_axis == 'x':
        for i in range(pix_resampled.shape[1]):
            imsave(os.path.join(RESAMPLE_FOLDER_x, patient, str(i) + '.png'), pix_resampled[:, i, :])

        # if save_axis == 'y':
        for i in range(pix_resampled.shape[2]):
            imsave(os.path.join(RESAMPLE_FOLDER_y, patient, str(i) + '.png'), pix_resampled[:, :, i])

        # Save X
        result_path = os.path.join(RESULT_FOLDER_x, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)
        CT2Xray_right_shift(os.path.join(RESAMPLE_FOLDER_x, patient), os.path.join(result_path, 'right.png'))
        CT2Xray_left_shift(os.path.join(RESAMPLE_FOLDER_x, patient), os.path.join(result_path, 'left.png'))
        CT2Xray_center_scale(os.path.join(RESAMPLE_FOLDER_x, patient), os.path.join(result_path, 'center.png'))
        normal(os.path.join(RESAMPLE_FOLDER_x, patient), os.path.join(result_path, 'normal.png'))

        # Save y
        result_path = os.path.join(RESULT_FOLDER_y, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)
        CT2Xray_right_shift(os.path.join(RESAMPLE_FOLDER_y, patient), os.path.join(result_path, 'right.png'))
        CT2Xray_left_shift(os.path.join(RESAMPLE_FOLDER_y, patient), os.path.join(result_path, 'left.png'))
        CT2Xray_center_scale(os.path.join(RESAMPLE_FOLDER_y, patient), os.path.join(result_path, 'center.png'))
        normal(os.path.join(RESAMPLE_FOLDER_y, patient), os.path.join(result_path, 'normal.png'))

        # Save z
        result_path = os.path.join(RESULT_FOLDER_z, patient)
        if not os.path.exists(result_path): os.mkdir(result_path)
        CT2Xray_right_shift(os.path.join(RESAMPLE_FOLDER_z, patient), os.path.join(result_path, 'right.png'))
        CT2Xray_left_shift(os.path.join(RESAMPLE_FOLDER_z, patient), os.path.join(result_path, 'left.png'))
        CT2Xray_center_scale(os.path.join(RESAMPLE_FOLDER_z, patient), os.path.join(result_path, 'center.png'))
        normal(os.path.join(RESAMPLE_FOLDER_z, patient), os.path.join(result_path, 'normal.png'))
