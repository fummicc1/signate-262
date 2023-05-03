#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import gc
import torch.distributed as dist
import pandas as pd
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt
from typing import List
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import json
from PIL.Image import Image
import PIL
from tqdm import tqdm
from PIL import ImageDraw
from ultralytics import YOLO
import subprocess
from wandb_callback import callbacks 


# ## Run

# In[13]:


mode = "all"
use_trained_weights = False


# In[ ]:


if mode == "all":
    kfolds = 4
    for i in range(3, kfolds):
        if i > 1:
            use_trained_weights = True
        data_path=f"/home/fummicc1/codes/signate/book_kfold{i}.yaml"
        if not use_trained_weights:
            weight_path="/home/fummicc1/codes/signate/yolov5x6.pt"
        else:
            weight_path="/home/fummicc1/codes/signate/yolo_all_weights-base.pt"
        out_path="/home/fummicc1/codes/signate/results/train_all_book_yolov5"
        command = f"cd yolov5 && python train.py --img 800 --epochs 30 --data {data_path} --batch-size 8 --weights {weight_path} --project {out_path} --device 0,1,2,3"
        subprocess.run(command, shell=True)
        break


# In[14]:


if mode == "modern":
    kfolds = 4
    for i in range(1, kfolds):
        if i > 1:
            use_trained_weights = True
        data_path=f"/home/fummicc1/codes/signate/modern_book_kfold{i}.yaml"
        if not use_trained_weights:
            weight_path="/home/fummicc1/codes/signate/yolov5x6.pt"
        else:
            weight_path="/home/fummicc1/codes/signate/yolo_modern_weights-base.pt"
        out_path="/home/fummicc1/codes/signate/results/train_modern_book_yolov5"
        command = f"cd yolov5 && python train.py --img 800 --epochs 30 --data {data_path} --batch-size 8 --weights {weight_path} --project {out_path} --device 0,1,2,3"
        subprocess.run(command, shell=True)
        break


# In[15]:


if mode == "old":
    kfolds = 4
    for i in range(1, kfolds):
        if i > 1:
            use_trained_weights = True
        data_path=f"/home/fummicc1/codes/signate/old_book_kfold{i}.yaml"
        if not use_trained_weights:
            weight_path="/home/fummicc1/codes/signate/yolov5x6.pt"
        else:
            weight_path="/home/fummicc1/codes/signate/yolo_old_weights-base.pt"        
        out_path="/home/fummicc1/codes/signate/results/train_old_book_yolov5"
        command = f"cd yolov5 && python train.py --img 800 --epochs 30 --data {data_path} --batch-size 8 --weights {weight_path} --project {out_path} --device 0,1,2,3"
        subprocess.run(command, shell=True)
        break

