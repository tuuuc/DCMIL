from telnetlib import NAMS
import openslide
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math
import glob

datasets = ['OV','BLCA','COAD','KIRC','LUAD','LUSC','THCA','HNSC','LIHC','BRCA','STAD','UCEC']
down_sample = 4
for dataset in datasets[-2:-1]:
	print(dataset)
	save_path = './wsi_thumbnail/{}'.format(dataset)
	if not os.path.exists(save_path): os.makedirs(save_path)

	for _file in glob.glob(f'./TCGA_{dataset}/*/*.svs'):
		name = os.path.basename(_file).split('.')[0]
		print(name)
		spath = "{}/{}.jpg".format(save_path,name)
		if os.path.exists(spath): continue
		# img_path = r"{}\{}".format(_path,name)
		slide = openslide.OpenSlide(_file)
		power = slide.properties['openslide.objective-power']

		if power not in ['20', '40']: continue
			
		[w,h] = slide.dimensions
		if power == '40':
			w = math.ceil(w/2)
			h = math.ceil(h/2)            
		w //= down_sample
		h //= down_sample
		print(w,h)
		try:
			slide_thumbnail = slide.get_thumbnail((w,h))
		except openslide.lowlevel.OpenSlideError:
			print('openslide.lowlevel.OpenSlideError')
			continue
		slide_thumbnail = np.array(slide_thumbnail)
		
		slide_thumbnail = cv2.cvtColor(slide_thumbnail,cv2.COLOR_RGB2BGR)
		cv2.imwrite(spath, slide_thumbnail)
