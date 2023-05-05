'''
	This script correctly formats the pickle output of running DensePose on a directory of images.
	Each frame_x_densepose.npy is stored in the same place as its corresponding image, frame_x.png.
'''
import os
import cv2
import glob
import tqdm
import torch
import numpy as np 

# Filepath to raw DensePose pickle output
outpath = '../UBC_Fashion_Dataset/detectron2/projects/DensePose/densepose.pkl'

# Convert pickle data to numpy arrays and save
data = torch.load(outpath)
for i in tqdm.tqdm(range(len(data))):
	dp = data[i]
	path = dp['file_name'] # path to original image
	dp_uv = dp['pred_densepose'][0].uv # uv coordinates
	h, w, c = cv2.imread(path).shape
	_, h_, w_ = dp_uv.shape
	(x1, y1, x2, y2) = dp['pred_boxes_XYXY'][0].int().numpy() # location of person
	y2, x2 = y1+h_, x1+w_
	dp_im = np.zeros((2, h, w))
	dp_im[:,y1:y2,x1:x2] = dp_uv.cpu().numpy()
	savepath = path.replace('.png', '_densepose.npy')
	np.save(savepath, dp_im)


