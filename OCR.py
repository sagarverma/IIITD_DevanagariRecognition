import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import math, random, glob, json, array

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from itertools import count

from gen_word_img import *

BATCH_SIZE = 1
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

fin = open('../dataset/synthetic_words/desc.json','rb')
datastore = json.load(fin)
fin.close()

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(288, 84)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

policy_net = DQN()
target_net = DQN()

policy_net.cuda()
target_net.cuda()

steps_done = 0

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    
    steps_done += 1
    if sample < eps_threshold:
        with torch.no_grad():
            result = policy_net(state)
            return Variable(result.max(1)[1]).cuda()
    else:
        result = []
        for i in range(BATCH_SIZE):
            result.append(random.randint(0,83))
        result = np.asarray(result,dtype=np.float32)
        return Variable(torch.tensor(result, dtype=torch.long)).cuda()

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
memory = ReplayMemory(10000)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(1)
    batch = Transition(*zip(*transitions))

    
    for s in batch.next_state:
        a = np.array(s)
    
    non_final_next_states = torch.cat([s for s in batch.next_state])
    non_final_next_states = non_final_next_states.view(non_final_next_states.size()[0],1,45,48)                                            
    
    state_batch = torch.cat(batch.state).cuda()
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values = target_net(non_final_next_states).detach().max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.cuda()
    
    loss = F.smooth_l1_loss(state_action_values.max(1)[0].unsqueeze(1),expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss

optimizer = optim.RMSprop(policy_net.parameters())
criterion = nn.MSELoss()       

def load_image(filename):
    img = Image.open(filename).convert('L')
    img.load()
    img = np.asarray(img, dtype=np.float32)
    img /= 255.
    img = np.expand_dims(img,axis=0)
    return img

def construct_image(samples):
    k, j = 0, 0
    char_imgs = []
    
    sample_huber_loss = 0.0
    sample_reward = 0.0
    
    while(j < samples.shape[3] - 48):
        images = samples[:, : ,:,j:j+48]
        
        state = Variable(torch.from_numpy(images).cuda())
        action = select_action(state)
        
        char_imgs.append(scribe_wrapper(datastore['abc'][action], "Devanagri 24", 45, 5, 0, 0))
        
        expected = []
        expected.append(np.expand_dims(char_imgs[k], axis=0))
        expected = np.asarray(expected, dtype = np.float32)
        expected = Variable(torch.from_numpy(expected).cuda())
        
        reward = criterion(state, expected)
        reward.backward()
        reward = reward / (45 * 48)
        
        if(reward < 2):
            j = j + 48
            if(j+48 > samples.shape[3]):
                next_state = state
            else:
                next_state = Variable(torch.from_numpy(samples[:, :, :, j:j+48]).cuda())
        else:
            next_state = state
            j += 48
        
        reward = torch.tensor([reward])
        
        memory.push(state, action, next_state, reward)
        hubber_loss = optimize_model()
        
        sample_huber_loss = hubber_loss.cpu().data.numpy()
        sample_reward = reward.cpu().data.numpy()
        k += 1
    if(k!=0):
        return sample_huber_loss / k, sample_reward / k
    else:
        return sample_huber_loss,sample_reward
def main():
    mean = [0.5657177752729754, 0.5381838567195789, 0.4972228365504561]
    std = [0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    
    DATASET_DIR = "../dataset/synthetic_words/data/*.png"
    img_files = glob.glob(DATASET_DIR)
    
    memory_image = []
    memory_image.append(load_image(img_files[0]))
        
    map = {}
    epoch_huber_loss = 0.0
    epoch_reward = 0.0
    
    for i in range(0,len(img_files)):
        #print i
        samples = []
        samples.append(load_image(img_files[i]))
        samples = np.asarray(samples, dtype=np.float32)
        
        huber_loss, reward = construct_image(samples)
        
        epoch_huber_loss += huber_loss
        epoch_reward += reward
        
        if i%1000 == 0:
            print 'Huber Loss ', epoch_huber_loss / (i+1), ' Reward ', epoch_reward / (i+1)
            
            
    torch.save(policy_net.state_dict(), '../weights/synthetic_dqn_model_84_classes.pth')

main()