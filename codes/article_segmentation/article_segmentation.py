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

masks_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_maps/'
images_dir = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_small_pngs/'
save_dir = 'checkpoint/'
    
class model(nn.Module):


    def __init__(self, n_class=2):
        super(model, self).__init__()
        
        self.f1 = nn.Sequential(
                        nn.Dropout2d(0.3),
                        nn.Conv2d(1, 32, 5, stride=1, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 16, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, stride=2, padding=0),
                        
                        nn.Dropout2d(0.3),
                        nn.Conv2d(16, 16, 5,stride=1, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 16, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, stride=2, padding=0),

                        nn.Dropout2d(0.3),
                        nn.Conv2d(16, 16, 5,stride=1, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(16, 16, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                )
        self.pool3 = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.f2 = nn.Sequential(        
                        nn.Dropout(0.5),
                        nn.Conv2d(16, 64, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 64, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                    )
        self.pool4 = nn.MaxPool2d(2, stride=2, padding=0)  # 1/8
                
        self.f3 = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Conv2d(64, 64, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(64, 128, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                    )
                    
        self.pool5 = nn.MaxPool2d(2, stride=2, padding=0)
                        
        self.f4 =   nn.Sequential(              
                        nn.Dropout2d(0.3),
                        nn.Conv2d(128, 128, 5,stride=1, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 256, 3,stride=1, padding=1),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, stride=2, padding=0),
                    )
                    
        self.up1 =  nn.Sequential( 
                            nn.ConvTranspose2d(256, 128, 2, stride=2),
                            nn.ReLU(inplace=True),
                    )
        self.up2 = nn.Sequential(
                            nn.ConvTranspose2d(128,64,2,stride=2),
                            nn.ReLU(inplace=True),
                    )
        self.up3 = nn.Sequential(
                            nn.ConvTranspose2d(64,16,2,stride=2),
                            nn.ReLU(inplace=True),
                    )
        self.up4 = nn.Sequential(
                            nn.ConvTranspose2d(16,16,4,stride=4),
                            nn.ReLU(inplace=True),
                    )
        
        self.refinement = nn.Sequential(
                            nn.Conv2d(16, 32, 5,stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, 5,stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 8, 1,stride=1, padding=0),
                            nn.Sigmoid(),
                            nn.Dropout2d(0.3),
                            nn.Conv2d(8, 32, 5,stride=1, padding=2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 16, 3,stride=1, padding=0),
                            nn.ReLU(inplace=True),
                        )
        
        self.classification = nn.Sequential(
                                nn.Conv2d(16,2,1,stride=1,padding=1),
                                nn.Sigmoid(),
                                nn.ConvTranspose2d(2,2,2,stride=2),
                                nn.ReLU(inplace=True),
                            )

    def forward(self, x):
        h = x
        f1 = self.f1(h)
        p3 = self.pool3(f1)
        f2 = self.f2(p3)
        p4 = self.pool4(f2)
        f3 = self.f3(p4)
        p5 = self.pool5(f3)
        f4 = self.f4(p5)
        u1 = self.up1(f4) + p5
        u2 = self.up2(u1)
        u2 = u2 + p4
        u3 = self.up3(u2) + p3
        u4 = self.up4(u3)
        
        r = self.refinement(u4)
        return self.classification(r)

def train():
    net = model()
    net.cuda()
    net.load_state_dict(torch.load(save_dir + 'model_at_18.pt'))
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    weights = []
    weights.append(1.8)
    weights.append(1.0)
    weights = torch.FloatTensor(weights).cuda()
    criterion = nn.NLLLoss(weight = weights)
    m = nn.LogSoftmax()
    #criterion = nn.BCELoss()
    mean = [0.5381838567195789]
    std = [0.2874722565279285]
    
   
    images = glob.glob(images_dir + "*.png")
    train_images = images[0:33000]
    test_images = images[33001:33976]

    TRAIN_SIZE = len(train_images)
    TEST_SIZE = len(test_images)
    BATCH_SIZE = 32
    
    for epoch in range(200):
        total_loss = 0
        samples = 0
        for i in range(0, TRAIN_SIZE, BATCH_SIZE):
            inputs = []
            masks = []
            
            for j in range(i, i+BATCH_SIZE):
                img = cv2.imread(images[j])
                img = cv2.resize(img, (256, 256))
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                img /= 255
                img = np.add(img, mean)
                img = np.divide(img, std)
                img = np.expand_dims(img,axis=0)
                inputs.append(img)
                mask = cv2.imread(masks_dir + images[i].split('/')[-1])
                mask = cv2.resize(mask, (256,256))
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask /= 255
                masks.append(mask)
            inputs = np.asarray(inputs, dtype=np.float32)
            masks = np.asarray(masks, dtype=np.long)
            inputs, masks = Variable(torch.from_numpy(inputs).cuda()), Variable(torch.from_numpy(masks).cuda())
            if len(masks) == BATCH_SIZE:
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                result = net(inputs)
                output = m(result)
                loss = criterion(output, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.data.cpu().numpy()
                samples += 1
                output = torch.max(output,1)[1].cpu().numpy()
                actual = masks.cpu().numpy()
                tp = 0
                fp = 0
                for k in range(BATCH_SIZE):
                    for l in range(256):
                        for n in range(256):
                            if(actual[k,l,n]==1):
                                if(output[k,l,n]==1):
                                    tp += 1
                            elif(output[k,l,n]==1):
                                fp += 1
                if(tp ==0):
                    print("Average precision for the batch:", 0)
                else:
                    print("Average precision for the batch:",float(tp)/float(tp+fp))
                torch.save(net.state_dict(), "checkpoint/model_at_" + str(epoch) + ".pt")
        total_loss = total_loss/float(samples)
        print("Loss after epochs is",[i,total_loss])
train()
    