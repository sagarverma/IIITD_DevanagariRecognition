from __future__ import print_function, division
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys 
from utils.dataloader import ImagePreloader

use_gpu = torch.cuda.is_available()

mean = 0.5
std = 1
lr = 0.01
momentum = 0.5
step_size = 30
gamma = 0.9
num_epochs = 30
data_dir = 'dataset/DevanagariHandwritten/'
train_csv = 'train.csv'
test_csv = 'test.csv'
num_classes = 46
batch_size = 128
weights_dir = 'weights/char_classifier/'
class_map = {'character_10_yna': 0,
 'character_11_taamatar': 1,
 'character_12_thaa': 2,
 'character_13_daa': 3,
 'character_14_dhaa': 4,
 'character_15_adna': 5,
 'character_16_tabala': 6,
 'character_17_tha': 7,
 'character_18_da': 8,
 'character_19_dha': 9,
 'character_1_ka': 10,
 'character_20_na': 11,
 'character_21_pa': 12,
 'character_22_pha': 13,
 'character_23_ba': 14,
 'character_24_bha': 15,
 'character_25_ma': 16,
 'character_26_yaw': 17,
 'character_27_ra': 18,
 'character_28_la': 19,
 'character_29_waw': 20,
 'character_2_kha': 21,
 'character_30_motosaw': 22,
 'character_31_petchiryakha': 23,
 'character_32_patalosaw': 24,
 'character_33_ha': 25,
 'character_34_chhya': 26,
 'character_35_tra': 27,
 'character_36_gya': 28,
 'character_3_ga': 29,
 'character_4_gha': 30,
 'character_5_kna': 31,
 'character_6_cha': 32,
 'character_7_chha': 33,
 'character_8_ja': 34,
 'character_9_jha': 35,
 'digit_0': 36,
 'digit_1': 37,
 'digit_2': 38,
 'digit_3': 39,
 'digit_4': 40,
 'digit_5': 41,
 'digit_6': 42,
 'digit_7': 43,
 'digit_8': 44,
 'digit_9': 45}
data_transforms= {
  'train': transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize([mean],[std])       
   ]),
   'test': transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize([mean], [std])
   ]),
  }


image_datasets = {'train': ImagePreloader(data_dir + 'Train/', data_dir + train_csv, class_map, data_transforms['train']), 
                    'test': ImagePreloader(data_dir + 'Test/', data_dir + test_csv, class_map, data_transforms['test'])}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes



######################################################################
        
def train_model(model, optimizer, num_epochs=2000):
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
                inputs, labels = data
                
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = F.nll_loss(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase=='train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                print ('##############################################################')
                print ("{} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))                
            else:
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)
                print (" {} loss = {}, acc = {},".format(phase, epoch_loss, epoch_acc))
                print ('##############################################################')


            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, weights_dir + 'weights_'+ file_name + '_lr_' + str(lr) + '_momentum_' + str(momentum) + '_step_size_' + \
                        str(step_size) + '_gamma_' + str(gamma) + '_num_classes_' + str(num_classes) + \
                        '_batch_size_' + str(batch_size) + '.pt')
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    
    return model

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':

    file_name = __file__.split('/')[-1].split('.')[0]
    print (file_name)
    
    model_conv = Net()
    print(model_conv)
    
    if use_gpu:
        model_conv = model_conv.cuda()

    optimizer_conv = optim.SGD(model_conv.parameters(), lr=lr, momentum=momentum)
    
    model_conv = train_model(model_conv, optimizer_conv, num_epochs=num_epochs)