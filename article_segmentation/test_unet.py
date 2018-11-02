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

def test():
    net = UNet(n_channels=3, n_classes=1)
    net.cuda()
    images_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_small_pngs/'
    masks_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_maps/'
    save_dir = 'unet_checkpoint/'
    images = glob.glob(images_dir+"*.png")
    pretrained_dict = torch.load(save_dir + 'model_at_73.pt')
    net.load_state_dict(pretrained_dict)
    net.eval()
    mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
    std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    for i in range(20):
        inputs = []
        img = cv2.imread(images[i])
        img = cv2.resize(img, (256, 256))
        mask = cv2.imread(masks_dir + images[i].split('/')[-1])
        mask = cv2.resize(mask, (256,256))
        cv2.imwrite("pdf/actual" + str(i) + ".jpg",mask)
        cv2.imwrite("pdf/input" + str(i) + ".jpg",img)
        img /= 255
        img = np.add(img, mean)
        img = np.divide(img, std)
        img = np.transpose(img, (2,0,1))
        inputs.append(img)
        inputs = np.asarray(inputs,dtype=np.float32)
        inputs = Variable(torch.from_numpy(inputs).cuda())
        result = net(inputs)
        masks_probs = F.sigmoid(result)
        output = (masks_probs > 0.5).float()[:,0]
        output = output*255
        output = np.array(output)
        output = np.squeeze(output,axis=0)
        cv2.imwrite("pdf/result" + str(i) + ".jpg",output)
test()