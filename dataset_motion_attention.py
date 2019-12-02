import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose, Normalize
import os
import os.path as osp
from opts import parse_opts_offline

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img_transform = Compose([
    Resize((224, 224), Image.BILINEAR),
    ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
num_frames = 16
opt = parse_opts_offline()

class Gesturedata(Dataset):
    """Jester gesture index and label"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.gesture_frame = pd.read_csv(csv_file, delimiter=' ')

    def __len__(self):
        return len(self.gesture_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        gesture_idx = self.gesture_frame.iloc[idx, 0] 
        gesture_idx = int(gesture_idx[22:])
        label = int(self.gesture_frame.iloc[idx, 2])
        num_items = int(self.gesture_frame.iloc[idx, 1])
        # create video tensor: n_frame x n_channel x h x w
        #dataset_dir = '/home/jic17022/gesture_recognition/dataset/jester/20bn-jester-v1/'
        dataset_dir = opt.video_path
        ges_dir = dataset_dir + str(gesture_idx)
        images = torch.zeros(1,3,num_items-1,224,224)
        #images = torch.zeros(3,num_frames,224,224
        attention_weights = torch.zeros(1,1,num_items-1)
        img_1_path = ges_dir + '/' + str(1).zfill(5) + '.jpg'
        img_1 =  img_transform(Image.open(img_1_path))
        img_1 = img_1.view(-1)
        for image_i in range(2, num_items+1):
            image_path = ges_dir + '/' + str(image_i).zfill(5) + '.jpg'
            img =  img_transform(Image.open(image_path))
            images[0,:,image_i-2,:,:]=img
            img = img.view(-1)
            attention_weights[0,0,image_i-2] = torch.norm(img-img_1)
            img_1 = img
            #images[:,image_i-1,:,:]=img
        images = nn.functional.interpolate(images,[num_frames-1,224,224],mode='trilinear',align_corners=True)
        images = torch.squeeze(images)
        attention_weights = nn.functional.interpolate(attention_weights,[num_frames-1],mode='linear',align_corners=True)
        attention_weights = attention_weights.squeeze()
        attention_weights = torch.nn.functional.softmax(attention_weights, 0)
        return  images, label, attention_weights
