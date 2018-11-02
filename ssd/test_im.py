from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import numpy as np
import argparse
import glob

def test_im():
    cfg = coco
    net = build_ssd('test', cfg['min_dim'], cfg['num_classes']) # initialize SSD
    net.load_state_dict(torch.load('/media/sagan/Drive2/sagar/staqu_ocr/codes/ssd.pytorch/weights/ssd300_COCO_5000.pth'))
    transform = SSDAugmentation(cfg['min_dim'],MEANS)
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True
    images = glob.glob('/media/sagan/Drive2/sagar/staqu_ocr/dataset/words_localized/*.png')
    for i in range(len(images)):
        if(i==0 or i==1):
            continue
        img = cv2.imread(images[i])
        x = cv2.imread(images[i])
        a = random.randint(0,img.shape[0])
        b = random.randint(0,img.shape[1])
        x = x[a:a + 300,b:b+300]
        img = img[a:a + 300,b:b+300]
        #img = np.asarray(img)
        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        x = x.cuda()
        y = net(x)      # forward pass
        detections = y.data
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        for j in range(200):
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            img = cv2.rectangle(img, (pt[0], pt[1]), (pt[2], pt[3]), (255,0,0), 2)
        break
        cv2.imwrite(str(i) + ".png",img)
if __name__ == '__main__':
    test_im()