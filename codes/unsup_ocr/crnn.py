from __future__ import print_function, division
import time
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

from warpctc_pytorch import CTCLoss

from utils import EnglishImagePreloader, StrLabelConverter, Averager, loadData

use_gpu = torch.cuda.is_available()
DEVICE = 0

# #Data statistics
# num_classes = DATA.rgb['num_classes']
# class_map = DATA.rgb['class_map']

#Training parameters
lr = 0.01
adam = False
adadelta = False
rmsprop = True
keep_ratio = True
batch_size = 512
imgH = 32
imgW = 100
nh = 256
lr = 0.01
beta1 = 0.5 #beta1 for adam
nclass = 27
nc = 1
num_epochs = 10
class_map = {chr(x): x-97 for x in range(97,97+26)}

data_dir = '../../datasets/english-words/'
train_csv = 'train_words_alpha.txt'
test_csv = 'test_words_alpha.txt'
weights_dir = '../../weights/'

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        # print (input.size())
        conv = self.cnn(input)
        # print (conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        # print (conv.size())
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)
        # print (output.size())
        return output

def train_model(model, criterion, optimizer, converter, num_epochs=2000):
    """
        Training model with given criterion, optimizer for num_epochs.
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_loss = []
    train_acc = []
    test_acc = []
    test_loss = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, words = data

                labels = torch.IntTensor(batch_size * 5)
                lengths = torch.IntTensor(batch_size)

                labels = Variable(labels)
                lengths = Variable(lengths)

                t, l = converter.encode(words)
                loadData(labels, t)
                loadData(lengths, l)


                if use_gpu:
                    inputs = Variable(inputs.cuda(DEVICE))
                    # labels = Variable(labels.cuda(DEVICE))
                    # lengths = Variable(lengths.cuda(DEVICE))
                else:
                    inputs, labels, lengths = Variable(inputs), Variable(labels), Variable(lengths)

                optimizer.zero_grad()
                outputs = model(inputs)
                # print (outputs.size())
                preds_size = Variable(torch.IntTensor([outputs.size(0)] * batch_size))
                loss = criterion(outputs, labels, preds_size, lengths)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

                _, preds = outputs.max(2)
                preds = preds.squeeze(1)
                preds = preds.transpose(1, 0).contiguous().view(-1).type(torch.IntTensor)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
            else:
                test_loss.append(epoch_loss)    # length_average (bool): normalize the loss by the total number of frames in the batch. If True, supersedes size_average (default: False)

                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_num_classes_' + str(nclass) + \
                        '_batch_size_' + str(batch_size) + '.pt')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    model = torch.load(weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_num_classes_' + str(nclass) + \
            '_batch_size_' + str(batch_size) + '.pt')

    return model



#Dataload and generator initialization
converter = StrLabelConverter(''.join(class_map.keys()) + ' ')
image_datasets = {'train': EnglishImagePreloader(data_dir + train_csv),
                    'test': EnglishImagePreloader(data_dir + test_csv)}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
file_name = __file__.split('/')[-1].split('.')[0]

#Create model and initialize/freeze weights
model_conv = CRNN(imgH, nc, nclass, nh)
print (model_conv)
model_conv = model_conv.cuda(DEVICE)

#Initialize optimizer and loss function
criterion = CTCLoss()
criterion = criterion.cuda()

if adam:
    optimizer = optim.Adam(model_conv.parameters(), lr=lr,
                           betas=(beta1, 0.999))
elif adadelta:
    optimizer = optim.Adadelta(model_conv.parameters(), lr=lr)
else:
    optimizer = optim.RMSprop(model_conv.parameters(), lr=lr)

#Train model
model_conv = train_model(model_conv, criterion, optimizer, converter, num_epochs=num_epochs)
