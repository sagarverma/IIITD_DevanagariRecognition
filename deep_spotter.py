import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import glob
from PIL import Image
import numpy as np

best_prec1 = 0
mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.m1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_bn(32, 32, 2),
            conv_bn(32,64,1),
            conv_bn(64,64,2),
            conv_bn( 64, 128, 1),
            conv_bn(128, 128, 1),
        )
        self.m2 = nn.Sequential(
            nn.MaxPool2d(2,stride = 2),
            conv_bn(128, 256, 1),
            conv_bn(256, 256, 1),
            conv_bn(256, 256, 1),
            conv_bn(256, 256, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.m3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            conv_dw(512,512,1),
            nn.Conv2d(512,168,1,1,0, bias=False)
        )

    def forward(self, x):
        x = self.m1(x)
        y = self.m2(x)
        x = torch.cat((x,x),1)
        print(x.size())
        print(y.size())
        z = torch.cat((x,y),1)
        r = self.m3(z)
        return r

def main():
    images = glob.glob('/media/sagan/Drive2/sagar/staqu_ocr/dataset/words_localized/*.png')
    model = Net()
    model = model.double().cuda()
    img = Image.open(images[0]).convert('RGB')
    img = img.resize((32, 32))
    img.load()
    img = np.asarray(img, dtype=np.float32)
    img /= 255.
    img = np.add(img, mean)
    img = np.divide(img, std)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img,0)
    inputs = Variable(torch.from_numpy(img).cuda())
    y = model(inputs)
    print(y.size())
main()