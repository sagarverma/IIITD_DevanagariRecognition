import utils.connected_components as cc
import utils.run_length_smoothing as rls
import utils.clean_page as clean
import utils.segmentation as seg
import utils.defaults as defaults
import utils.arg as arg
import cv2

from os import listdir

max_size=0
min_size=10
count = 1

imgs = listdir('dataset/newspaper_cropped/')

for img_name in imgs:
    img = cv2.imread('dataset/newspaper_cropped/' + img_name, 0)
    ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    segmented_image = seg.segment_image(thresh)
    segmented_image = segmented_image[:,:,2]
    components = cc.get_connected_components(thresh)
    for component in components:
        if min_size > 0 and cc.area_bb(component)**0.5<min_size: continue
        if max_size > 0 and cc.area_bb(component)**0.5>max_size: continue
        #a = area_nz(component,img)
        #if a<min_size: continue
        #if a>max_size: continue
        (ys,xs)=component[:2]
        ys_min = max(ys.start-5, 0)
        ys_max = ys.stop+5
        xs_min = max(xs.start-5, 0)
        xs_max = xs.stop+5
        patch = img[ys_min:ys_max, xs_min:xs_max]
        cv2.imwrite('dataset/newspaper_words/' + str(count).zfill(10) + '.png', patch)
        count += 1
    