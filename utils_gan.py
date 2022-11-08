# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import argparse
import numpy as np
import torch
from torch import nn
import cv2
#import imageio

def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label




def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
     
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))  # 查找bmp文件，存入列表中
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))

    # print(data)
    return data


def make_data(data, label, data_dir):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """

    savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))
    if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)



def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        # flatten=True 
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)





def input_setup(opt,data_dir,index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path

    
    data = prepare_data(dataset=data_dir)


    sub_input_sequence = []
    sub_label_sequence = []
    padding = int(abs(opt.img_size - opt.label_size) / 2)  # 6


    for i in range(len(data)):
        # input_, label_ = preprocess(data[i], opt.scale)
        input_ = (imread(data[i]) - 127.5) / 127.5
        label_ = input_

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        
        for x in range(0, h - opt.img_size + 1, opt.stride):
            for y in range(0, w - opt.img_size + 1, opt.stride):
                sub_input = input_[x:x + opt.img_size, y:y + opt.img_size]  # [33 x 33]
                
                sub_label = label_[x + padding:x + padding + opt.label_size,
                            y + padding:y + padding + opt.label_size]  # [21 x 21]

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  
    arrlabel = np.asarray(sub_label_sequence)  
    # print(arrdata.shape)
    make_data(arrdata, arrlabel, data_dir)  

    # if not opt.is_train:
    #     print(nx, ny)
    #     print(h_real, w_real)
    #     return nx, ny, h_real, w_real


def imsave(image, path):
    return scipy.misc.imsave(path, image)



def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return (img * 127.5 + 127.5)


