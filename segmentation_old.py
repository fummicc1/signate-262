#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pathlib

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils as smp_utils
import albumentations as albu
import torch.nn as nn


# In[ ]:


os.chdir("/home/fummicc1/codes/signate/")


# In[ ]:


TRAIN_DIR = pathlib.Path('./train')
VAL_DIR = pathlib.Path('./val')
TEST_DIR = pathlib.Path('./test')


# In[ ]:


x_train_dir = TRAIN_DIR / "old" / "images"
y_train_dir = TRAIN_DIR / "old" / "masks"

x_valid_dir = VAL_DIR / "old" / "images"
y_valid_dir = VAL_DIR / "old" / "masks"

x_test_dir = TEST_DIR / "old" / "images"
y_test_dir = TEST_DIR / "old" / "masks"


# In[ ]:


def get_training_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=960, min_width=960, always_apply=True, border_mode=0),
        albu.RandomCrop(height=960, width=960, always_apply=True)
    ]
    return albu.Compose(train_transform)
    
def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(1216, 1216)
    ]
    return albu.Compose(test_transform)


def to_tensor(x: np.ndarray, **kargs):
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float)

def get_preprocessing(preprocessing_fn, has_mask: bool = True):
    _transform = [
        albu.Lambda(image=preprocessing_fn) if preprocessing_fn is not None else albu.Compose([]),
        albu.Lambda(image=to_tensor, mask=to_tensor) if has_mask else albu.Lambda(image=to_tensor)
    ]
    return albu.Compose(_transform)


# In[ ]:


# 可視化用の関数
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# In[ ]:


# パラメータ
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = "softmax"
CLASS_MAP = {  
  0: "0_background",
  1: "2_handwritten",
  2: "3_typography",
  3: "4_illustration",
  4: "5_stamp",
}
CLASSES = CLASS_MAP.values()
DEVICE = 'cuda'


# In[ ]:


class MyDataset(Dataset):
    
    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
        preprocessing=None
    ) -> None:
        self.ids = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, id) for id in self.ids]
        self.masks = [os.path.join(masks_dir, id) for id in self.ids]
        
        # クラス名の文字列
        self.class_values = CLASS_MAP.keys()
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
    def __len__(self) -> int:
        return len(self.ids)
        
    # 3. 学習用データ(image)と特徴(mask)を返す__getitem__メソッドを作成
    def __getitem__(self, i):
        
        # データの読み込み
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not os.path.exists(self.masks[i]):            
            # 前処理関数の適用
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image = sample['image']
            return image, None
        mask = cv2.imread(self.masks[i], 0) // 25
        
        # 学習対象のクラス(例えば、'car')のみを抽出
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('uint')
        # print("mask.shape", mask.shape)
        
        # augmentation関数の適用
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # 前処理関数の適用
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask


# In[ ]:


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# In[ ]:


# SMPを用いて学習済みモデルを取得(アーキテクチャはFPN)
model = smp.UnetPlusPlus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
model = model.to(device=DEVICE).type(torch.float)
model = nn.DataParallel(model).to(device=DEVICE).type(torch.float)


# In[ ]:


# 損失関数
loss = smp_utils.losses.DiceLoss(activation="softmax2d")

# 評価関数
metrics = [
    smp_utils.metrics.IoU(threshold=0.5)
]

# 最適化関数
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])


# In[ ]:


from torch.utils.data import DataLoader


# データセットのインスタンスを作成
train_dataset = MyDataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(None),
)

# データローダーの作成
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=12)


# In[ ]:


from torch.utils.data import DataLoader

# データセットのインスタンスを作成
val_dataset = MyDataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(None),
)

# データローダーの作成
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=12)


# In[ ]:


train_epoch = smp_utils.train.TrainEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp_utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True
)


# In[ ]:


epochs = 20
max_score = 0

if not pathlib.Path("best_model_old.pth").exists():
    for epoch in range(1, epochs + 1):
        print(f"\n Epoch: {epoch}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        # 評価関数の値が更新されたらモデルを保存
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, './best_model_old.pth')
            print('Model saved!')

        # エポック10以降は学習率(learning rate)を下げる      
        if epoch == 10:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')


# ## Classify

# In[ ]:


from PIL.Image import Image, open as im_open
import PIL.Image

# 可視化用の画像を取得するデータセットを作成(Augmentationなし)
test_dataset_vis = MyDataset(
    x_valid_dir, y_valid_dir, 
)

# 可視化用の関数
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap="gray")
    plt.figure(figsize=(8, 4))
    data = np.array(list(CLASS_MAP.keys()))
    data *= 25
    colors = []
    for d in data:
        d /= 255
        colors.append((d, d, d))
    plt.bar(x=CLASS_MAP.values(), height=data, color=colors)
    plt.show()


# In[ ]:


# 1. 学習モデルの読み込み
best_model = torch.load(pathlib.Path("best_model_old.pth").as_posix())

# 2. 推論用のデータセット、データローダーの作成
valid_dataset = MyDataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(None),
)
valid_dataloader = DataLoader(valid_dataset)

n_data = 2
for i in range(n_data):
    n = np.random.choice(len(valid_dataset))
    # n = 100
    # n = i

    # 3. 新規データの取得
    image_vis = test_dataset_vis[n][0].astype(np.uint8)
    image, gt_mask = valid_dataset[n]
    gt_mask = gt_mask.squeeze()    
    
    # 3. 新規データの推論
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.module.predict(x_tensor)    
    pr_mask: torch.Tensor = pr_mask.squeeze(dim=0)
    _, pr_mask = torch.max(pr_mask, dim=0)
    pr_mask = torch.squeeze(pr_mask, dim=0)
    pr_mask = pr_mask.cpu().detach().numpy()  
    pr_mask *= 25

    _, gt_mask = torch.max(gt_mask, dim=0)
    gt_mask = gt_mask.cpu().detach().numpy()    
    gt_mask *= 25
    
    print(gt_mask.shape, pr_mask.shape)

    # 4. 可視化
    visualize(
        image=image_vis, 
        ground_truth_mask=gt_mask, 
        predicted_mask=pr_mask
    )


# In[ ]:


# 1. 学習モデルの読み込み
best_model = torch.load(pathlib.Path("best_model_old.pth").as_posix())

# 2. 推論用のデータセット、データローダーの作成
valid_dataset = MyDataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(None, has_mask=False),
)
test_dataloader = DataLoader(valid_dataset)

n_data = len(valid_dataset) # 確認するデータの数
for i in range(0):
    # n = np.random.choice(len(test_dataset))
    # n = 100
    n = i

    # 3. 新規データの取得
    image, _ = valid_dataset[n]
    
    # 3. 新規データの推論
    x_tensor = image.to(DEVICE).unsqueeze(0)
    pr_mask = best_model.module.predict(x_tensor)    
    pr_mask: torch.Tensor = pr_mask.squeeze(dim=0)
    _, pr_mask = torch.max(pr_mask, dim=0)
    pr_mask = pr_mask.cpu().detach().numpy()
    pr_mask *= 25

    out_dir = TEST_DIR / "old" / "predicted"
    out_dir.mkdir(exist_ok=True)
    plt.imsave(out_dir / list(x_test_dir.iterdir())[i].name, pr_mask, cmap="gray")

