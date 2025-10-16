#!/usr/bin/env python
# -*- coding:utf-8 -*-
from fileinput import filename
from tkinter import filedialog
import openslide
import numpy as np
import time
import scipy.io as scio
import os
import cv2
from openslide.deepzoom import DeepZoomGenerator
import warnings
import math
from skimage.exposure import match_histograms
from tqdm import tqdm
from PIL import Image

# os.environ["PATH"] = "" + ":" + os.environ["PATH"]

def img_path(dirname):
    filter1 = [".svs"]
    img_path = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            ext = os.path.splitext(filename)[1]
            if ext in filter1:
                img_path.append(os.path.join(maindir, filename))
    return img_path

def extract_roi(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    s2 = cv2.blur(s, (5, 5))
    _, saturation_thresholded_mask = cv2.threshold(s2, 15, 255, cv2.THRESH_BINARY)
 
    roi = saturation_thresholded_mask / 255
    return roi

def get_patch(object_path, path):
    
    fild_id_downloaded = img_path(object_path)  # svs_address
    num = fild_id_downloaded.__len__()
    print(num)
    img_size = [128, 256, 512]
    
    for n in range(num):
        file_dir = fild_id_downloaded[n]
        file_name = file_dir.split('.')[0][-23:]
     
        if file_name[-3:-1] != 'TS': continue
        if int(file_name[-10:-8]) < 10:
            spath = [f'{path}/tumor/{m}/{file_name}/' for m in ['5x','10x','20x']]
            loc = path + '/tumor/location'
        else:
            spath = [f'{path}/normal/{m}/{file_name}/' for m in ['5x','10x','20x']]
            loc = path + '/normal/location'
        if os.path.exists(loc+f'/{file_name}.mat'):
            print(file_name)
            continue           
        print(file_name)        
        os.makedirs(loc, exist_ok=True)
        try:
            slide = openslide.OpenSlide(file_dir)
        except openslide.lowlevel.OpenSlideError:
            print('exception')
            continue                                 
         
        data_128 = DeepZoomGenerator(slide, tile_size=img_size[0], overlap=0, limit_bounds=False)
        data_256 = DeepZoomGenerator(slide, tile_size=img_size[1], overlap=0, limit_bounds=False)
        data_512 = DeepZoomGenerator(slide, tile_size=img_size[2], overlap=0, limit_bounds=False)
             
        [w, h] = slide.level_dimensions[0]      #最高倍下的宽高
        level = data_128.level_count-1
        power = slide.properties['openslide.objective-power']
      
        if power not in ['20', '40']: continue
         
        if power == '40':
            w = math.ceil(w/2)
            h = math.ceil(h/2)
            level -= 1
             
        # print(w, h)
 
        num_h = int(h / img_size[2])
        num_w = int(w / img_size[2])
 
        location = np.zeros((num_h, num_w))
        seq = 1
 
        for i in tqdm(range(num_h)):
            for j in range(num_w):
                try:
                    img = [np.array(data_128.get_tile(level-2, (j, i)))]
                except Exception:
                    # print('exception')
                    continue
                roi = extract_roi(img[0])
                rate = np.sum(roi) / (img_size[0] * img_size[0])
 
                if rate > 0.5:
                    try:
                        img.append(np.array(data_256.get_tile(level-1, (j, i))))
                        img.append(np.array(data_512.get_tile(level, (j, i))))
                    except Exception:
                        continue
                    for p in spath:
                        os.makedirs(p,exist_ok=True)
                    for k in range(len(img)):
                        matched = Image.fromarray(img[k])
                        matched.save('%s%d%s' % (spath[k], seq, '.jpg'))                 
                     
                    location[i, j] = 1
                    seq = seq + 1
 
        if seq > 1:
            dataNew = loc + f'/{file_name}.mat'
            scio.savemat(dataNew, {'location': location})

    return


if __name__ == '__main__':
    warnings.filterwarnings('error')
    time_start = time.time()

    datasets = ['KIRC','LUAD','LUSC','STAD','THCA','HNSC','LIHC','BRCA','BLCA','OV','UCEC','COAD']

    for dataset in datasets:
        print(dataset)
        main_path = f'./TCGA/{dataset}_TS'
        save_path = f'./{dataset}_ms'
        patches = get_patch(main_path, save_path)

