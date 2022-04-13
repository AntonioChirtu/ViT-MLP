import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from matplotlib.image import imread
from PIL import Image
import numpy as np


class RemoteSensingDataset(Dataset):
    def __init__(self, dataset_dir, type, split, transform=None):
        self.img_labels = []
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.images = []
        self.type = type
        self.split = split

        if type == 'train':
            for class_dir in os.listdir(self.dataset_dir):
                for idx, img in enumerate(os.listdir(os.path.join(self.dataset_dir, class_dir))):
                    num_train = len(os.listdir(os.path.join(self.dataset_dir, class_dir)))
                    split_idx = int(np.floor(split * num_train))
                    if idx < split_idx:
                        self.img_labels.append(int(class_dir))
                        self.images.append(img)
        else:
            for class_dir in os.listdir(self.dataset_dir):
                for idx, img in enumerate(os.listdir(os.path.join(self.dataset_dir, class_dir))):
                    num_train = len(os.listdir(os.path.join(self.dataset_dir, class_dir)))
                    split_idx = int(np.floor(split * num_train))
                    if idx >= split_idx:
                        self.img_labels.append(int(class_dir))
                        self.images.append(img)

        self.img_labels = np.asarray(self.img_labels)

    def __len__(self):
        if self.type == 'train':
            l = len(self.images)
        if self.type == 'test':
            l = len(self.images)
        return l

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_dir, str(self.img_labels[idx]), self.images[idx])
        image = Image.open(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.asarray(label))
        return image, label
