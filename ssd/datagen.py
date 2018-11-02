'''Load image/class/box from a annotation file.

The annotation file is organized as:
    image_name #obj xmin ymin xmax ymax class_index ..
'''
from __future__ import print_function

import os
import sys
import os.path
import glob

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from encoder import DataEncoder
from PIL import Image, ImageOps

training_data_path = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/words_localized/'

import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class ListDataset(data.Dataset):
    img_size = 300

    def __init__(self, root, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.train = train
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        lines = glob.glob(training_data_path + "*.png")
        lines.sort(key=alphanum_key)
        self.num_samples = len(lines)
        
        files = glob.glob(training_data_path + "*.npy")

        for file in files:
            self.fnames.append(file.split('.')[0] + ".png")
            boxes = np.load(file)
            num_objs = boxes.shape[0]
            target = []
            label = []
            for i in range(num_objs):
                xmin = boxes[i,0]
                ymin = boxes[i,1]
                xmax = boxes[i,2]
                ymax = boxes[i,3]
                target.append([float(xmin),float(ymin),float(xmax),float(ymax),1])
            self.boxes.append(torch.Tensor(target))

    def __getitem__(self, idx):
        '''Load a image, and encode its bbox locations and class labels.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Data augmentation while training.
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w,h = img.size
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def __len__(self):
        return self.num_samples