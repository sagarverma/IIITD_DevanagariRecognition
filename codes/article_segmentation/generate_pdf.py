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
from fpdf import FPDF

def test():
    actuals = sorted(glob.glob("pdf/actual*"))
    result = sorted(glob.glob("pdf/result*"))
    inputs = sorted(glob.glob("pdf/input*"))
    pdf = FPDF()
    x = 0
    y = 0
    for i in range(len(actuals)):
        pdf.add_page()
        #print(inputs[i])
        print(actuals[i])
        #print(result[i])
        pdf.image(inputs[i])
        pdf.image(actuals[i])
        pdf.image(result[i])
    pdf.output("result.pdf", "F")
test()
        
