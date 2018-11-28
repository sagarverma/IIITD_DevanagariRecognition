from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class SegmentationImageDataset(Dataset):
    """Article Segmentation dataset. - Dainik Bhaskar"""

    def __init__(self, image_list, mask_dir, root_dir, transform=None):
        """
        Args:
            mask_dir (string): Directory with all the masks.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mask_dir = mask_dir
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        img = cv2.imread(self.image_list[idx])
        shape = img.shape
        img = img/255.
        img = np.add(img, mean)
        img = np.divide(img, std)
        img = cv2.resize(img,(256,256))
        img = np.transpose(img,(2,1,0))
    
        with open(self.mask_dir + self.image_list[idx].split('/')[-1].split('.')[0] + '.pkl', 'r') as f:
            x = pickle.load(f)
        marker = np.zeros((shape[0],shape[1]))
        for k in x:
            box = k
            marker[box[0]:box[2],box[1]:box[3]] = x[k]
        marker = cv2.resize(marker,(256,256))
        marker = marker.astype(int)
        sample = {'image': img, 'marker': marker}

        return sample