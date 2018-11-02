import cv2
import glob
import pickle
import random
import numpy as np
import operator
import json

directory = '/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_pngs/*.png'
images = glob.glob(directory)

with open('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_png_bbox_maps_adjacency_map.pkl') as f:
    dicti = pickle.load(f)
with open('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_png_bbox_maps.pkl') as f:
    boxes = pickle.load(f)

for i in range(len(images)):
    try:
        img = cv2.imread(images[i])
        mapped_boxes = dicti[images[i].split('/')[-1]]['mapped_bboxes']
        adjacency_map = dicti[images[i].split('/')[-1]]['adjacency_map']
    except:
        continue
    if img is None:
        continue
    classes = {}
    max_local = 0
    cmap = {}
    
    areas = {}
    
    for k in mapped_boxes:
        box = mapped_boxes[k]
        areas[k] = (box[3] - box[1])*(box[2]-box[0])
    
    dec = sorted(areas.items(), key=operator.itemgetter(1),reverse=True)
        
    for j in dec:
        marked = np.zeros((100))
        box = mapped_boxes[j[0]]
        
        try:
            adjacent_boxes = adjacency_map[j[0]]
        except:
            adjacent_boxes = []
            max_local += 1
            classes[j[0]] = max_local
            continue
        
        for k in adjacent_boxes:
            if k in classes.keys():
                marked[classes[k]] = 1

        for k in range(len(marked)):
            if(marked[k]==0):
                classes[j[0]] = k
                break
    for j in dec:
        box = mapped_boxes[j[0]]
        cmap[(box[0],box[2],box[1],box[3])] = classes[j[0]]
    with open('/media/sagan/Drive2/sagar/staqu_ocr/dataset/danik_bhaskar_class_maps/' + images[i].split('/')[-1].split('.')[0] + '.pkl', 'w') as outfile:  
        pickle.dump(cmap, outfile)
    print(i)