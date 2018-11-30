from __future__ import print_function, division
import time
import os
import shutil
import csv
import random

import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

from warpctc_pytorch import CTCLoss

from utils.dataloader import EnglishImagePreloader2, text_loader


class_map = {chr(x): x-97 for x in range(97,97+26)}
class_map[' '] = 26
inv_class_map = {x-97: chr(x) for x in range(97,97+26)}
inv_class_map[26] = ' '

class Environment(object):

    def __init__(self, text_file, word_len, sim_speed=0.0001):
        self.text_file = text_file
        self.word_len = word_len
        self.words = self._get_words()
        self.sim_speed = sim_speed

        self.board, self.word = text_loader(random.sample(self.words, 1)[0], self.word_len, False)
        self.board_size = self.board.shape
        self.slate = [' ' for x in range(self.word_len)]

        self.done = False

    def _get_words(self):
        r = csv.reader(open(self.text_file, 'r'))
        words = []
        for row in r:
            if len(row[0]) <= self.word_len:
                words.append(row[0])
        return words

    def _get_rewards(self):
        slate_image, _ = text_loader(''.join(self.slate), self.word_len, False)
        # reward = 1 - mse(self.board.flatten(), slate_image.flatten())
        reward = ssim(self.board, slate_image)
        if reward == 1:
            self.done = True
        return reward

    def step(self, write_head, write_char):
        self.slate[write_head] = inv_class_map[write_char]
        slate_image, _ = text_loader(''.join(self.slate), self.word_len, False)
        img_diff = self.board - slate_image
        img_diff = cv2.resize(img_diff, (120,32))
        img_diff = np.asarray([img_diff])
        return img_diff, self._get_rewards(), self.done

    def render(self):
        slate_image, _ = text_loader(''.join(self.slate), self.word_len, False)
        img = np.concatenate((self.board, slate_image), axis=0)

        plt.imshow(img, cmap='gray')
        plt.pause(self.sim_speed)
        plt.draw()


    def reset(self):
        self.board, self.word = text_loader(random.sample(self.words, 1)[0], self.word_len, False)
        self.board_size = self.board.shape
        self.slate = [' ' for x in range(self.word_len)]

        self.done = False

        slate_image, _ = text_loader(''.join(self.slate), self.word_len, False)
        img_diff = self.board - slate_image
        img_diff = cv2.resize(img_diff, (120,32))
        img_diff = np.asarray([img_diff])
        return img_diff
