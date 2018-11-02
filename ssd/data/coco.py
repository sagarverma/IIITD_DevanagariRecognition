from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import glob

COCO_ROOT = osp.join(HOME, 'data/coco/')
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'PythonAPI'
INSTANCES_SET = 'instances_{}.json'

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

class COCOAnnotationTransform(object):
    def __init__(self):
        print("do nothing")

    def __call__(self, target, width, height):
        scale = np.array([width, height, width, height])
        res = []
        for bbox in target:
            label_idx = 0
            final_box = list(np.array(bbox)/scale)
            final_box.append(label_idx)
            res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            
        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):


    def __init__(self, root, transform=None, target_transform=COCOAnnotationTransform()):
        
        self.root = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/words_localized/'
        self.ids = glob.glob(self.root + "*.png")
        self.ids.sort(key=alphanum_key)
        for i in range(len(self.ids)):
            a = self.ids[i]
            self.ids[i] = a.split('/')[-1].split('.')[0]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):

        img_id = self.ids[index]
        ann_ids = self.root + img_id + ".npy"

        target = np.load(ann_ids)
        for i in range(target.shape[0]):
            for j in range(target[i].shape[0]):
                target[i,j] = float(target[i,j])
        
        img = cv2.imread(self.root + img_id + ".png")
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        img_id = self.ids[index]
        path = self.root + img_id + ".png"
        return cv2.imread(path, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        img_id = self.ids[index]
        ann_ids = self.root + img_id + ".npy"
        boxes = np.load(ann_ids)
        for i in range(boxes.shape[0]):
            for j in range(boxes[i].shape[0]):
                boxes[i,j] = float(boxes[i,j])
        return np.load(boxes)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
