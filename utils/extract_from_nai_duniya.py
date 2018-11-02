from pdf2image import convert_from_path
import pdfplumber
import cv2
import numpy as np 
from math import ceil, floor
from os import listdir
import json
from random import shuffle

PDF_DIR = '../dataset/nai_duniya/'
pdfs = listdir(PDF_DIR)
pdfs.sort()

white_spaces = ['\t','U+0008', 'U+0009', 'U+000A', 'U+000B', 'U+000C', 'U+000D', 'U+0020', 'U+0085', 'U+00A0', 'U+1680', 'U+2000', 'U+2001', 'U+2002', 'U+2003', 'U+2004', 'U+2005', 'U+2006', 'U+2007', 'U+2008', 'U+2009', 'U+200A', 'U+2028', 'U+2029', 'U+202F', 'U+205F', 'U+3000', ' ']

frm = 0
to = 1000

fout = open('../dataset/nai_duniya_words_clean_' + str(frm) + '_' + str(to) + '.json','wb')

out = {}

done = {}

word_no = 0
for pdf in pdfs[frm:to]:
    print pdf
    image = convert_from_path(PDF_DIR + pdf, fmt='jpg')
    image = image.pop()
    img = np.asarray(image)
    
    pdf = pdfplumber.open(PDF_DIR + pdf)
    first_page = pdf.pages[0]
    words = first_page.extract_words()
    pdf_height = float(first_page.height)
    pdf_width = float(first_page.width)

    img_height = img.shape[0]
    img_width = img.shape[1]

    height_scale = img_height/ pdf_height  
    width_scale = img_width/ pdf_width 
    
    for word in words:
        xmin = int(floor(float(word['x0']) * width_scale))
        xmax = int(ceil(float(word['x1']) * width_scale))
        ymin = int(floor(float(word['top']) * height_scale))
        ymax = int(ceil(float(word['bottom']) * height_scale))

        patch = img[ymin:ymax, xmin:xmax, :]

        width = patch.shape[1]
        chars = len(word['text'])
        word_text = word['text']
        
        if word_text + '_' + str(patch.shape[0]) + '_' + str(patch.shape[1]) not in done  and patch.shape[0] > 0 and patch.shape[1] > 0:
            done[word_text + '_' + str(patch.shape[0]) + '_' + str(patch.shape[1])] = 1
            cv2.imwrite('../dataset/nai_duniya_words_clean/' + str(word_no).zfill(10) + '.png', patch)                    
            out[str(word_no).zfill(10)] = word_text
            word_no += 1



json.dump(out, fout)

fout.close()