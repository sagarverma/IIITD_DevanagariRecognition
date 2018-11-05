import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import glob
import cv2
from PIL import Image
from sklearn.metrics import average_precision_score
from dataloader import ImageSegmentationDataset
from unet_parts import *
import random
import operator
import pickle

save_dir = 'unet_final_checkpoint/'

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

def train():
    
    net = UNet(n_channels=3, n_classes=13)
    save_dir = 'unet_final_checkpoint/'
    net.cuda(1)
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    
    segmentation_dataset = ImageSegmentationDataset(mask_dir='/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_class_maps/'
        ,root_dir= '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_pngs/')
    
    TRAIN_SIZE = 0.75*len(segmentation_dataset)
    TEST_SIZE = 0.25*len(segmentation_dataset)
    BATCH_SIZE = 32
    
    for epoch in range(200):
        total_loss = 0
        samples = 0
        for i in range(0,TRAIN_SIZE,BATCH_SIZE):
            inputs = []
            masks = []
            max_local = 0
            for j in range(i,i+BATCH_SIZE):
                sample = segmentation_dataset[j]
                img = sample['image']
                marker = sample['marker']
                if img is None or marker is None:
                    continue
                mask = np.zeros((256,256,13))
                for k in range(marker.shape[0]):
                    for l in range(marker.shape[1]):
                        mask[k,l,mark[k,l]] = 1
                masks.append(mask)
                inputs.append(img)
            inputs = np.asarray(inputs, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            inputs, masks = Variable(torch.from_numpy(inputs).cuda(1)), Variable(torch.from_numpy(masks).cuda(1))
            if len(masks) == BATCH_SIZE:
                # forward + backward + optimize
                result = net(inputs)
                masks_probs = F.sigmoid(result)
                masks_probs_flat = masks_probs.view(-1)
                true_masks_flat = masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                samples += 1

        total_loss = total_loss/float(samples)
        print ('##############################################################')
        print ("{} Train_loss = {}".format(epoch, total_loss))
        
        total_loss = 0
        samples = 0
        for i in range(TRAIN_SIZE + 1,TRAIN_SIZE + TEST_SIZE,BATCH_SIZE):
            inputs = []
            masks = []
            max_local = 0
            for j in range(i,i+BATCH_SIZE):
                sample = segmentation_dataset[j]
                img = sample['image']
                marker = sample['marker']
                if img is None or marker is None:
                    continue
                mask = np.zeros((256,256,13))
                for k in range(marker.shape[0]):
                    for l in range(marker.shape[1]):
                        mask[k,l,mark[k,l]] = 1
                masks.append(mask)
                inputs.append(img)

            inputs = np.asarray(inputs, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.float32)
            inputs, masks = Variable(torch.from_numpy(inputs).cuda(1)), Variable(torch.from_numpy(masks).cuda(1))
            if len(masks) == BATCH_SIZE:
                # forward + backward + optimize
                result = net(inputs)
                masks_probs = F.sigmoid(result)
                masks_probs_flat = masks_probs.view(-1)
                true_masks_flat = masks.view(-1)
                loss = criterion(masks_probs_flat, true_masks_flat)
                total_loss += loss.item()
        total_loss = total_loss/float(samples)
        print ('##############################################################')
        print ("{} Test_loss = {}".format(epoch, total_loss))


train()