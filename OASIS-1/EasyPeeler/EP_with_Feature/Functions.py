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
import pickle

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_train_pair(data_path, filename1, filename2, feat_to_use):
    # Load images and labels
    image1 = pkload(os.path.join(data_path, filename1, 'slice_norm_{}.pkl'.format(feat_to_use)))
    # image1 = nim1.get_fdata()
    image1 = np.array(image1, dtype='float32')

    image2 = pkload(os.path.join(data_path, filename2, 'slice_norm_{}.pkl'.format(feat_to_use)))
    # image2 = nim2.get_fdata()
    image2 = np.array(image2, dtype='float32')
    
    data_path1 = '/Datasets/Dataset_OASIS/OASIS_AffineData/'
    
    nim3 = nib.load(os.path.join(data_path1, filename1, 'slice_norm.nii.gz'))
    image3 = nim3.get_fdata()[:,:,0]
    image3 = np.array(image3, dtype='float32')

    nim4 = nib.load(os.path.join(data_path1, filename2, 'slice_norm.nii.gz'))
    image4 = nim4.get_fdata()[:,:,0]
    image4 = np.array(image4, dtype='float32')


    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)

    return image1, image2, image3, image4

class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, img_file=None, trainingset = 1, feat_to_use = 0):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file),dtype='str')
        self.feat_to_use = feat_to_use
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
            # print(len(self.names))
            # print(len(self.filename))
            assert len(self.filename) == 40200, "Oh no! # of images != 40200."
        
        else:
             assert 0==1, print('TrainDataset Invalid!')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_40, fix_40, mov_img, fix_img = load_train_pair(self.data_path, self.filename[index][0], self.filename[index][1], self.feat_to_use)
        return mov_40, fix_40, mov_img, fix_img

class ValidationDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, img_file=None, feat_to_use = 0):
        'Initialization'
        super(ValidationDataset, self).__init__()

        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file),dtype='str')
        self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
        self.filename = self.zip_filename_1 + self.zip_filename_2
        self.feat_to_use = feat_to_use
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        
        mov_40, fix_40, mov_img, fix_img, mov_lab, fix_lab = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1], self.feat_to_use)
        return self.filename[index][0],self.filename[index][1], mov_40, fix_40, mov_img, fix_img, mov_lab, fix_lab
def load_validation_pair(data_path, filename1, filename2, feat_to_use):
    # Load images and labels
    image1 = pkload(os.path.join(data_path, filename1, 'slice_norm_{}.pkl'.format(feat_to_use)))
    # image1 = nim1.get_fdata()
    image1 = np.array(image1, dtype='float32')

    image2 = pkload(os.path.join(data_path, filename2, 'slice_norm_{}.pkl'.format(feat_to_use)))
    # image2 = nim2.get_fdata()
    image2 = np.array(image2, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, filename1, 'slice_seg24.nii.gz'))
    image5 = nim5.get_fdata()[:,:,0]
    image5 = np.array(image5, dtype='float32')
    # image5 = image5 / 35.0
    nim6 = nib.load(os.path.join(data_path, filename2, 'slice_seg24.nii.gz'))
    image6 = nim6.get_fdata()[:,:,0]
    image6 = np.array(image6, dtype='float32') # 0 - 35 -》 0- 1
    # image6 = image6 / 35.0
    
    
    data_path1 = '/Datasets/Dataset_OASIS/OASIS_AffineData/'
    
    nim3 = nib.load(os.path.join(data_path1, filename1, 'slice_norm.nii.gz'))
    image3 = nim3.get_fdata()[:,:,0]
    image3 = np.array(image3, dtype='float32')

    nim4 = nib.load(os.path.join(data_path1, filename2, 'slice_norm.nii.gz'))
    image4 = nim4.get_fdata()[:,:,0]
    image4 = np.array(image4, dtype='float32')
    
    
    image5 = np.reshape(image5, (1,) + image5.shape)
    image6 = np.reshape(image6, (1,) + image6.shape)
    image3 = np.reshape(image3, (1,) + image3.shape)
    image4 = np.reshape(image4, (1,) + image4.shape)
    
    return image1, image2, image3, image4, image5, image6
    