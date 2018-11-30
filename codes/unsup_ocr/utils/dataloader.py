import sys
import os, csv, random, array, collections

import cv2
from PIL import Image
import numpy as np

import cffi_wrapper as cp
import cairocffi
from trimmers import horztrim

import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize

import torch
import torch.utils.data as data
from torch.autograd import Variable

from torchvision.transforms import functional


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

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

def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(images_list):

    classes = {}
    class_id = 0
    for image in images_list:
        if image[1] not in classes:
            classes[image[1]] = class_id
            class_id += 1

    return classes.keys(), classes

def make_dataset(dir, images_list, class_to_idx):
    images = []

    for image in images_list:
        images.append((dir + image[0], int(image[1])))

    return images

def make_sequence_dataset(dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        images = []
        for image in sequence[0]:
            images.append(dir + image)

        sequences.append([images, int(sequence[1])])

    return sequences

def make_bimode_sequence_dataset(rgb_dir, flow_dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        rgb_images = []
        flow_images = []

        for image in sequence[0]:
            rgb_images.append(rgb_dir + image)
            flow_images.append(flow_dir + image)

        sequences.append([rgb_images, flow_images, int(sequence[1])])

    return sequences

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def text_loader(target, word_len, resize=True):
    word = target + ' ' * (word_len - len(target))
    img = scribe_wrapper(word, "Devanagri 24", 45, 5, 0, 0)
    if resize:
        img = cv2.resize(img, (120, 32))
    img = 255 - img
    img = img / 255.0
    # return np.asarray([img]).astype(np.float32), word
    return img, word

def word_transform(char_map, word):
    labels = []
    for w in word_map:
        labels.append(char_map[w])
    return labels

class StrLabelConverter(object):
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

class Averager(object):
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

class ImagePreloader(data.Dataset):

    def __init__(self, root, csv_file, class_map, transform=None, target_transform=None,
                 loader=default_loader):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0],row[1]])


        random.shuffle(images_list)
        classes, class_to_idx = class_map.keys(), class_map
        imgs = make_dataset(root, images_list, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



class EnglishImagePreloader(data.Dataset):
    def __init__(self, csv_file, loader=text_loader):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append(row[0])
        random.shuffle(images_list)

        self.imgs = images_list
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        target = self.imgs[index]
        img, word = self.loader(target)

        return img, word

    def __len__(self):
        return len(self.imgs)

class EnglishImagePreloader2(data.Dataset):
    def __init__(self, csv_file, loader=text_loader):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append(row[0])
        random.shuffle(images_list)

        self.imgs = images_list
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        target = self.imgs[index]
        img, word = self.loader(target)

        return img, img

    def __len__(self):
        return len(self.imgs)
