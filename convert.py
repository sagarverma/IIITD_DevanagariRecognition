from os import listdir
import xml.etree.ElementTree as ET
import cv2
from scipy.ndimage.measurements import label
import numpy as np
from random import shuffle
import json
import matplotlib.pyplot as plt
fin = open('dataset/digital_words_clean_0_1000.json', 'rb')
data = json.load(fin)
out_data = {}
chars = {}
words = {}
for d in data.keys():
    text = data[d]
    chs = list(text)
    for ch in chs:
        if ch not in chars:
            chars[ch] = 1
        else:
            chars[ch] += 1
    text = text.replace('\t', '')
    out_data[d] = text
    
    if text not in words:
        words[text] = 1
    else:
        words[text] += 1
    
fin.close()
fout = open('dataset/digital_words_train_test.json','wb')
dataset = []

for d in data.keys():
    if '\t' not in data[d]:
        dataset.append([d, data[d]])
    
shuffle(dataset)
train = dataset[:int(len(dataset)*0.7)]
test = dataset[int(len(dataset)*0.7):]

print len(train), len(test)
out = {}

out['abc'] = ''.join(chars.keys())
out['train'] = [{"text": t[1], "name": t[0] + '.png'} for t in train]
out['test'] = [{"text": t[1], "name": t[0] + '.png'} for t in test]
json.dump(out, fout)
fout.close()