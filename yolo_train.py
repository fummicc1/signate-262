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

data_path="/workspace/modern_book.yaml"
weight_path="/workspace/yolov8x.pt"
out_path="/workspace/results/train_modern_book"
model = YOLO(weight_path)
for event,func in callbacks.items():
    model.add_callback(event,func)
model.train(
    data=data_path,
    epochs=30,
    imgsz=1600,
    batch=4,
    lr0=0.01,
    project=out_path,
    device="0,1,2,3"
)
# command=f'yolo detect train model={weight_path} data={data_path} epochs=30 imgsz=1600 batch=8 lr0="0.01" project="{out_path}" device="0,1,2,3"'
# subprocess.run(command, shell=True)


data_path="/workspace/old_book.yaml"
weight_path="/workspace/yolov8x.pt"
out_path="/workspace/results/train_old_book"
model = YOLO(weight_path)
for event,func in callbacks.items():
    model.add_callback(event,func)
model.train(
    data=data_path,
    epochs=30,
    imgsz=1600,
    batch=4,
    lr0=0.01,
    project=out_path,
    device="0,1,2,3"
)
# command=f'yolo detect train model={weight_path} data={data_path} epochs=30 imgsz=1600 batch=8 lr0="0.01" project="{out_path}" device="0,1,2,3"'
# subprocess.run(command, shell=True)
