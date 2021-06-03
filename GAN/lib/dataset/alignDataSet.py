# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from X2CT.GAN.lib.dataset.baseDataSet import Base_DataSet
from X2CT.GAN.lib.dataset.utils import *
import h5py
import numpy as np
import os
import torch
import cv2
#
# class AlignDataSet(Base_DataSet):
#   '''
#   DataSet For unaligned data
#   '''
#   def __init__(self, opt):
#     super(AlignDataSet, self).__init__()
#     self.opt = opt
#     self.ext = '.h5'
#     self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
#     self.dataset_paths = sorted(self.dataset_paths)
#     self.dataset_size = len(self.dataset_paths)
#     self.dir_root = self.get_data_path
#     self.data_augmentation = self.opt.data_augmentation(opt)
#
#   @property
#   def name(self):
#     return 'AlignDataSet'
#
#   @property
#   def get_data_path(self):
#     path = os.path.join(self.opt.dataroot)
#     return path
#
#   @property
#   def num_samples(self):
#     return self.dataset_size
#
#   def get_image_path(self, root, index_name):
#     img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
#     assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
#     return img_path
#
#   # def load_file(self, file_path):
#   #   hdf5 = h5py.File(file_path, 'r')
#   #   ct_data = np.asarray(hdf5['ct'])
#   #   x_ray1 = np.asarray(hdf5['xray1'])
#   #   x_ray1 = np.expand_dims(x_ray1, 0)
#   #   hdf5.close()
#   #   return ct_data, x_ray1
#   #
#   # '''
#   # generate batch
#   # '''
#   # def pull_item(self, item):
#   #   file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
#   #   ct_data, x_ray1 = self.load_file(file_path)
#   #
#   #   # Data Augmentation
#   #   ct, xray1 = self.data_augmentation([ct_data, x_ray1])
#   #
#   #   return ct, xray1, file_path
#
#
#
#   def load_file(self, file_path):
#       '''
#
#       :param file_path: dir_root/dataset_paths/file.npy  ---> CT
#       :return:
#       '''
#       ct_name = os.path.join(file_path)
#       xray_name = os.path.join(file_path.replace('3d_numpy_array', 'xray_image').replace('.npy','.png').replace('CT_3D_', 'normal_'))
#       ct_data = np.load(ct_name)
#       xray_data = cv2.imread(xray_name, 0)
#       x_ray1 = np.expand_dims(xray_data, 0)
#
#       return ct_data, x_ray1
#
#
#
#   '''
#   generate batch
#   '''
#   def pull_item(self, item):
#     file_path = self.dataset_paths[item] #self.get_image_path(self.dir_root, self.dataset_paths[item])
#     ct_data, x_ray1 = self.load_file(file_path)
#     # assert ct_data.shape[0] == x_ray1.shape[1] and ct_data.shape[1] == x_ray1.shape[2]
#     # Data Augmentation
#     ct, xray1 = self.data_augmentation([ct_data, x_ray1])
#
#     return ct, xray1, file_path
#
#
# from torch.utils.data import Dataset
# class My_Align_DataSet(Dataset):
#   '''
#   Base DataSet
#   '''
#   @property
#   def name(self):
#     return 'AlignDataSet'
#
#   def __init__(self, opt):
#     self.opt = opt
#     self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
#     self.dataset_paths = sorted(self.dataset_paths)
#     self.dataset_size = len(self.dataset_paths)
#     self.dir_root = self.get_data_path
#     self.data_augmentation = self.opt.data_augmentation(opt)
#
#   def get_data_path(self):
#     path = os.path.join(self.opt.dataroot)
#     return path
#
#   def get_image_path(self, root, index_name):
#     img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
#     assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
#     return img_path
#
#   def load_file(self, file_path):
#       '''
#
#       :param file_path: dir_root/dataset_paths/file.npy  ---> CT
#       :return:
#       '''
#       ct_name = os.path.join(file_path)
#       xray_name = os.path.join(file_path.replace('3d_numpy_array', 'xray_image').replace('.npy','.png').replace('CT_3D_', 'normal_'))
#       seg_name = os.path.join(file_path.replace('3d_numpy_array', 'seg_image').replace('CT_3D_Patient', ''))
#       ct_data = np.load(ct_name)
#       xray_data = cv2.imread(xray_name, 0)
#       x_ray1 = np.expand_dims(xray_data, 0)
#       seg_data = np.expand_dims(np.load(seg_name), 0)
#       seg_data[seg_data<0.8] = 0
#       seg_data[seg_data>=0.8] = 1
#
#       return ct_data, x_ray1, seg_data
#
#
#   def __getitem__(self, item):
#     file_path = self.dataset_paths[item] #self.get_image_path(self.dir_root, self.dataset_paths[item])
#     ct_data, x_ray1, seg_data = self.load_file(file_path)
#     # print(file_path, ct_data.shape, x_ray1.shape)
#     # assert ct_data.shape[0] == x_ray1.shape[1] and ct_data.shape[1] == x_ray1.shape[2]
#     # Data Augmentation
#     ct, xray1 = self.data_augmentation([ct_data, x_ray1])
#     # segmentation_map
#     seg = torch.Tensor(seg_data)
#     return ct, xray1, seg, file_path
#
#   def __len__(self):
#     return self.dataset_size
#


class AlignDataSet(Base_DataSet):
  '''
  DataSet For unaligned data
  '''
  def __init__(self, opt):
    super(AlignDataSet, self).__init__()
    self.opt = opt
    self.ext = '.h5'
    self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
    self.dataset_paths = sorted(self.dataset_paths)
    self.dataset_size = len(self.dataset_paths)
    self.dir_root = self.get_data_path
    self.data_augmentation = self.opt.data_augmentation(opt)

  @property
  def name(self):
    return 'AlignDataSet'

  @property
  def get_data_path(self):
    path = os.path.join(self.opt.dataroot)
    return path

  @property
  def num_samples(self):
    return self.dataset_size

  def get_image_path(self, root, index_name):
    img_path = os.path.join(root, index_name, 'ct_xray_data'+self.ext)
    assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
    return img_path

  def load_file(self, file_path):
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray1 = np.expand_dims(x_ray1, 0)
    hdf5.close()
    return ct_data, x_ray1

  '''
  generate batch
  '''
  def pull_item(self, item):
    file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
    ct_data, x_ray1 = self.load_file(file_path)

    # Data Augmentation
    ct, xray1 = self.data_augmentation([ct_data, x_ray1])

    return ct, xray1, file_path







