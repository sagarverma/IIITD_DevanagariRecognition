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
from torch.utils.data import DataLoader
from dataloader import SegmentationImageDataset
from unet_parts import *
import random
import operator
import pickle

save_dir = 'unet_final_checkpoint/'

# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions

def iou(pred, target):
    ious = []
    unique, counts = np.unique(target, return_counts=True)
    for cls in range(13):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return float(correct) / float(total)

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
    save_dir = '/media/sagan/Drive2/sagar/ocr_art_seg/weights/article_segmentation/unet_article_seg_checkpoint/'
    net.cuda(1)
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    
    image_list = glob.glob('/media/sagan/Drive2/sagar/ocr_art_seg/dataset/danik_bhaskar_pngs/' + '*.png') 
    
    train_dataset = SegmentationImageDataset(image_list[0:30000], mask_dir='/media/sagan/Drive2/sagar/ocr_art_seg/dataset/danik_bhaskar_class_maps/'
        ,root_dir= '/media/sagan/Drive2/sagar/ocr_art_seg/dataset/danik_bhaskar_pngs/')
    
    test_dataset = SegmentationImageDataset(image_list[30000:], mask_dir='/media/sagan/Drive2/sagar/ocr_art_seg/dataset/danik_bhaskar_class_maps/'
        ,root_dir= '/media/sagan/Drive2/sagar/ocr_art_seg/dataset/danik_bhaskar_pngs/')
    
    train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=32,shuffle=True, num_workers=4)

    BATCH_SIZE = 32
    
    for epoch in range(200):
        total_loss = 0
        samples = 0
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs = []
            masks = []
            images_batch, marker_batch = sample_batched['image'], sample_batched['marker']
            
            batch_size = len(images_batch)
            for j in range(batch_size):
                max_local = 0
                img = images_batch[j]
                marker = marker_batch[j]
                if img is None or marker is None:
                    continue
                mask = np.zeros((256,256,13))
                for k in range(marker.shape[0]):
                    for l in range(marker.shape[1]):
                        val = int(marker[k,l].cpu().data)
                        mask[k,l,marker[k,l]] = 1
                masks.append(mask)
                inputs.append(img)
            
            inputs = [t.numpy() for t in inputs]

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
            torch.save(net.state_dict(), save_dir + 'model_at_' + str(epoch) + '.pt')
        total_ious = []
        pixel_accs = []
        _,pred = masks_probs.max(1)
        pred = pred.cpu().data.numpy().reshape(BATCH_SIZE, 256, 256)
        _,expec = masks.max(3)
        target = expec.cpu().data.numpy().reshape(BATCH_SIZE, 256, 256)
        
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))
        #  Calculate average IoU
        total_ious = np.array(total_ious).T  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()
        print("Mean Train IOU for each class after epoch {} ",epoch)
        print(ious)
        print(" Train Precision after epoch {} is {}",epoch,pixel_accs )
        print(pixel_accs)

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
                        val = int(marker[k,l].cpu().data)
                        mask[k,l,marker[k,l]] = 1
                masks.append(mask)
                inputs.append(img)

            inputs = [t.numpy() for t in inputs]

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
                samples += 1
        
        total_ious = []
        pixel_accs = []
        _,pred = masks_probs.max(1)
        pred = pred.cpu().data.numpy().reshape(BATCH_SIZE, 256, 256)
        _,expec = masks.max(3)
        target = expec.cpu().data.numpy().reshape(BATCH_SIZE, 256, 256)
        
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))
        
        total_ious = np.array(total_ious).T  # n_class * val_len
        ious = np.nanmean(total_ious, axis=1)
        pixel_accs = np.array(pixel_accs).mean()
        print("Mean Test IOU for each class after epoch {} ",epoch)
        print(ious)
        print(" Test Precision after epoch {} is {}",epoch,pixel_accs )
        print(pixel_accs)
                
        total_loss = total_loss/float(samples)
        print ('##############################################################')
        print ("{} Test_loss = {}".format(epoch, total_loss))

train()