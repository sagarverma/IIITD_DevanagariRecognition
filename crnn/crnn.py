import os
import time
import pickle
import cv2
import random
import array

import numpy as np

import cffi_wrapper as cp
import cairocffi
from trimmers import horztrim

import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

def scribe(text, font_style,
           width, height,
           movex, movey,
           twist):

    format = cairocffi.FORMAT_A8
    width = cairocffi.ImageSurface.format_stride_for_width(format, width)
    data = array.array('b', [0] * (height * width))
    surface = cairocffi.ImageSurface(format, width, height, data, width)
    context = cairocffi.Context(surface)
    context.translate(movex, movey)
    context.rotate(twist)

    layout = cp.gobject_ref(cp.pangocairo.pango_cairo_create_layout(context._pointer))
    cp.pango.pango_layout_set_text(layout, text.encode('utf8'), -1)

    font_desc = cp.pango.pango_font_description_from_string(font_style.encode('utf8'))
    cp.pango.pango_layout_set_font_description(layout, font_desc)

    cp.pangocairo.pango_cairo_update_layout(context._pointer, layout)
    cp.pangocairo.pango_cairo_show_layout(context._pointer, layout)

    return np.frombuffer(data, dtype=np.uint8).reshape((height, width))


def scribe_wrapper(text, font_style,
                   height, hbuffer, vbuffer,
                   twist):
    """
    Calcuates the image dimensions from given text and then renders it.
    :param text: Unicode Text
    :param font_style: "Gautami Bold 40", "Mangal Bold Italic 32" etc.
    :param height: Total height of the image.
    :param hbuffer: horizontal margin
    :param vbuffer: vertical margin
    :param twist:  rotation
    :return: an numpy array
    """

    lines = text.split('\n')
    n_lines = len(lines)
    n_letters = max(len(line) for line in lines)
    line_ht = height / (n_lines+1)
    letter_wd = .7 * line_ht
    width = int(round((n_letters+2) * letter_wd))
    
    #print("Lines:", n_lines, "Letters:", n_letters)
    #print("Line Height:", line_ht, " Letter Width:", letter_wd)
    #print("\nFont:{}\nWidth, Height:{} Area={}\nMargins:{}\nRotation:{}".format(
    #    font_style, (width, height), width*height, hbuffer, vbuffer, twist))

    return scribe(text, font_style, width, height, hbuffer, vbuffer, twist)

class ImagePreloader(data.Dataset):

    def __init__(self, train_set, class_map, imgH, imgW, transform=None, target_transform=None):

        self.words = train_set
        self.classes = len(class_map.keys())
        self.class_to_idx = class_map
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        word = self.words[index]

        img = scribe_wrapper(word, "Devanagri 24", 45, 5, 0, 0)
        img = horztrim(img, 3)
        img = 255 - img
        img = cv2.resize(img, (imgW, imgH))
        img /= 255
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, word

    def __len__(self):
        return len(self.words)
    
    
class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assureRatio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

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
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

fin = open('../dataset/parallel/unique_words.pkl','rb')
data = pickle.load(fin)
fin.close()

hindi_words = data[0][:20000]
class_map = data[1]    
batchSize = 256
imgH = 32
imgW = 100
nh = 256
niter = 25
lr = 0.01
beta1 = 0.5 #beta1 for adam

model_path = ''
chars = ''.join(class_map.keys())
adam = False
adadelta = False
rmsprop = True
keep_ratio = True
random_sample = True

experiment = '../weights/synthetic_word_crnn/'
displayInterval = 50
n_test_disp = 10
valInterval = 50
saveInterval = 50

nclass = len(chars) + 1
nc = 1


train_set = hindi_words[: int(0.7 * len(hindi_words))]
test_set = hindi_words[int(0.7 * len(hindi_words)): ]

train_dataset = ImagePreloader(train_set, class_map, imgH, imgW)
test_dataset = ImagePreloader(test_set, class_map, imgH, imgW)

train_loader = torch.utils.data.DataLoader( 
    train_dataset, batch_size=batchSize,
    shuffle=True, num_workers=4)

converter = strLabelConverter(chars)
criterion = CTCLoss()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
crnn = CRNN(imgH, nc, nclass, nh)
crnn.cuda()
crnn.apply(weights_init)

loss_avg = averager()

if adam:
    optimizer = optim.Adam(crnn.parameters(), lr=lr,
                           betas=(beta1, 0.999))
elif adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    
    


crnn = torch.nn.DataParallel(crnn, device_ids=range(1))
criterion = criterion.cuda()

def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    
    image = torch.FloatTensor(batchSize, 1, imgH, imgH)
    image = image.cuda()
    text = torch.IntTensor(batchSize * 5)
    length = torch.IntTensor(batchSize)
    
    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    loadData(text, t)
    loadData(length, l)
    
    image = image.view(image.size(0), 1, image.size(1), image.size(2))
    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=batchSize, num_workers=4)
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        
        image = torch.FloatTensor(batchSize, 1, imgH, imgH)
        text = torch.IntTensor(batchSize * 5)
        length = torch.IntTensor(batchSize)

        image = Variable(image)
        text = Variable(text)
        length = Variable(length)

    
        loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        loadData(text, t)
        loadData(length, l)

        image = image.view(image.size(0), 1, image.size(1), image.size(2))
        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze(1)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    
    
print 'Start Training'   
for epoch in range(niter):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        print i, displayInterval
        if i % displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(experiment, epoch, i))