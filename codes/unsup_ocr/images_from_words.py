import os
import cv2
import csv
import random
import array

import numpy as np

import cffi_wrapper as cp
import cairocffi
from trimmers import horztrim

import pickle

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



r = csv.reader(open('../../datasets/english-words/words_alpha.txt', 'r'))

# "dichlorodiphenyltrichloroethane"
# "d                              "
images = {}
for row in r:
    word = row[0]
    img = scribe_wrapper("d                              ", "Devanagri 24", 45, 5, 0, 0)
    print img.shape, img.min(), img.max()
    img = horztrim(img, 3)
    img = 255 - img
    img = img / 255
    print img.shape, img.max(), img.min(), img.mean()
    # img = img * 255
    # cv2.imwrite('../../datasets/english-words/pngs/' + word + '.png', img)
    images[word] = img
    break

# fout = open('../../datasets/english-words/images_alpha.pkl', 'wb')
# pickle.dump(images, fout)
