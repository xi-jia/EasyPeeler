"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join
import itertools

def load_train_pair(data_path, filename1, filename2):
    # Load images
    nim1 = nib.load(os.path.join(data_path, filename1, 'slice_norm.nii.gz'))
    image1 = nim1.get_fdata()[:,:,0]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, filename2, 'slice_norm.nii.gz'))
    image2 = nim2.get_fdata()[:,:,0]
    image2 = np.array(image2, dtype='float32')
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    
    return image1, image2

class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, img_file=None, trainingset = 1):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file),dtype='str')
        if trainingset == 1:
            self.filename = list(zip(self.names[:-1], self.names[1:]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 2:
            self.filename = list(zip(self.names[1:], self.names[:-1]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 3:
            self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
            self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
            self.filename = self.zip_filename_1 + self.zip_filename_2
            assert len(self.filename) == 400, "Oh no! # of images != 400."
        elif trainingset == 4:
            self.filename = list(itertools.permutations(self.names, 2))
            assert len(self.filename) == 40200, "Oh no! # of images != 40200."
        
        else:
             assert 0==1, print('TrainDataset Invalid!')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_train_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return  mov_img, fix_img

class ValidationDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, img_file=None):
        'Initialization'
        super(ValidationDataset, self).__init__()
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file),dtype='str')
        self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
        self.filename = self.zip_filename_1 + self.zip_filename_2
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A, img_B, label_A, label_B = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return self.filename[index][0],self.filename[index][1], img_A, img_B, label_A, label_B
def load_validation_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, filename1, 'slice_norm.nii.gz'))
    image1 = nim1.get_fdata()[:,:,0]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, filename2, 'slice_norm.nii.gz'))
    image2 = nim2.get_fdata()[:,:,0]
    image2 = np.array(image2, dtype='float32')
    
    
    nim5 = nib.load(os.path.join(data_path, filename1, 'slice_seg24.nii.gz'))
    seg_1 = nim5.get_fdata()[:,:,0]
    seg_1 = np.array(seg_1, dtype='float32')
    nim6 = nib.load(os.path.join(data_path, filename2, 'slice_seg24.nii.gz'))
    seg_2 = nim6.get_fdata()[:,:,0]
    seg_2 = np.array(seg_2, dtype='float32')
    
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    seg_1 = np.reshape(seg_1, (1,) + seg_1.shape)
    seg_2 = np.reshape(seg_2, (1,) + seg_2.shape)
    return image1, image2, seg_1, seg_2
    