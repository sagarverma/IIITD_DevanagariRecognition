import glob
import xml.etree.cElementTree as ET
import numpy as np
import cv2
direc = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/words_localized/'

files = glob.glob(direc + '*.npy')

for file in files:
    name = "/media/sagan/Drive2/sagar/staqu_ocr/xml/" + file.split('/')[-1].split('.')[0] + ".xml"
    image = direc + file.split('/')[-1].split('.')[0] + '.png'
    root = ET.Element("annotation")
    boxes = np.load(file)
    print(boxes)
    k = 0
    for box in boxes:
        k += 1
        object1 = ET.SubElement(root,'object')
        name = ET.SubElement(object1,'name')
        name.text = str('text')
        bndbox = ET.SubElement(object1,'bndbox')
        xmin = ET.SubElement(bndbox,'xmin')
        xmin.text = str(int(box[0]))
        ymin = ET.SubElement(bndbox,'ymin')
        ymin.text = str(int(box[1]))
        xmax = ET.SubElement(bndbox,'xmax')
        xmax.text = str(int(box[2]))
        ymax = ET.SubElement(bndbox,'ymax')
        ymax.text = str(int(box[3]))
    folder = ET.SubElement(root,'folder')
    folder.text = direc
    filename = ET.SubElement(root,'filename')
    filename.text = file.split('/')[-1].split('.')[0] + '.png'
    img = cv2.imread(image)
    size = ET.SubElement(root,'size')
    width = ET.SubElement(size,'width')
    width.text = str(img.shape[0])
    print(img.shape[0])
    height = ET.SubElement(size,'height')
    height.text = str(img.shape[1])
    print(img.shape[1])
    depth = ET.SubElement(size,'depth')
    depth.text = str(3)
    mydata = ET.tostring(root)
    myfile = open("/media/sagan/Drive2/sagar/staqu_ocr/dataset/xml/" + file.split('/')[-1].split('.')[0] + ".xml",'w')  
    myfile.write(mydata)
