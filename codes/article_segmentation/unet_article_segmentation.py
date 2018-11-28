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
from unet_parts import *
import random
import operator
import pickle

masks_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_class_maps/'
images_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_pngs/'
save_dir = 'unet_final_checkpoint/'

with open('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_png_bbox_maps_adjacency_map.pkl') as f:
    dicti = pickle.load(f)

with open('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_png_bbox_maps.pkl') as f:
    boxes = pickle.load(f)

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
    
    
    mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
    std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
   
    images = glob.glob(images_dir + "*.png")
    train_images = images[0:33000]
    test_images = images[33001:33976]

    TRAIN_SIZE = len(train_images)
    TEST_SIZE = len(test_images)
    BATCH_SIZE = 32
    
    for epoch in range(200):
        total_loss = 0
        samples = 0
        for i in range(0,TRAIN_SIZE,BATCH_SIZE):
            inputs = []
            masks = []
            max_local = 0
            for j in range(i,i+BATCH_SIZE):
                try:
                    img = cv2.imread(images[j])
                    img /= 255
                    img = np.add(img, mean)
                    img = np.divide(img, std)
                except:
                    continue
                if img is None:
                    continue
                with open(masks_dir + images[j].split('/')[-1].split('.')[0] + '.pkl', 'r') as f:
                    x = pickle.load(f)
                marker = np.zeros((img.shape[0],img.shape[1]))
                for k in x:
                    box = k
                    marker[box[0]:box[2],box[1]:box[3]] = x[k]
                img = cv2.resize(img,(256,256))
                img = np.transpose(img,(2,1,0))
                marker = cv2.resize(marker,(256,256))
                mark = marker.astype(int)
                mask = np.zeros((256,256,13))
                for k in range(mark.shape[0]):
                    for l in range(mark.shape[1]):
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
                actual = masks.cpu().numpy()
                output = (masks_probs > 0.5).float()[:,0]
                torch.save(net.state_dict(), "unet_final_checkpoint/model_at_" + str(epoch) + ".pt")
        total_loss = total_loss/float(samples)
        print ('##############################################################')
        print ("{} loss = {}".format(epoch, total_loss))
train()