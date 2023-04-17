#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


# os.chdir("/workspace")


# ## Run

# In[ ]:


mode = ""
use_yolo_pretrained = False


# In[ ]:


if mode == "":    
    data_path="/home/fummicc1/codes/signate/modern_book.yaml"
    if not use_yolo_pretrained:
        weight_path="/home/fummicc1/codes/signate/yolov5x6.pt"
    else:
        weight_path="/home/fummicc1/codes/signate/yolo_modern_weights-base.pt"
    out_path="/home/fummicc1/codes/signate/results/train_modern_book_yolov5"
    command = f"cd yolov5 && python train.py --img 1200 --epochs 30  --data {data_path} --weights {weight_path} --project {out_path} --device 0,1,2,3"
    # --batch-size 8
    subprocess.run(command, shell=True)


# In[ ]:


if mode == "":
    data_path="/home/fummicc1/codes/signate/old_book.yaml"
    
    if not use_yolo_pretrained:
        weight_path="/home/fummicc1/codes/signate/yolov5x6.pt"
    else:
        weight_path="/home/fummicc1/codes/signate/yolo_old_weights-base.pt"
    out_path="/home/fummicc1/codes/signate/results/train_old_book_yolov5"
    command = f"cd yolov5 && python train.py --img 1200 --epochs 30 --data {data_path} --weights {weight_path} --project {out_path} --device 0,1,2,3"
    subprocess.run(command, shell=True)

