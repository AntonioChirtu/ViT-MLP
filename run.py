from transformers import ViTFeatureExtractor, ViTForImageClassification, BatchFeature
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
import torchmetrics
from torchvision.datasets import CIFAR10

from img_model import ImageClassifier, MLP

from PIL import Image
import requests
import os
from tqdm import tqdm
from torch import optim
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint


class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


class ViTFeatureExtractorTransforms:
    def __init__(self, model_name_or_path):
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        transform = []

        if feature_extractor.do_resize:
            transform.append(Resize(feature_extractor.size))

        transform.append(ToTensor())

        if feature_extractor.do_normalize:
            transform.append(Normalize(feature_extractor.image_mean, feature_extractor.image_std))

        self.transform = Compose(transform)

    def __call__(self, x):
        return self.transform(x)


if __name__ == '__main__':
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 10
    batch_size = 8
    num_workers = 2
    max_epochs = 4

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # The directory with the test set contains 30% of data (used to train)
    trainset = datasets.ImageFolder(root="/home/antonio/PycharmProjects/ViT-MLP/Sydney/test",
                                    transform=transform_train)
    testset = datasets.ImageFolder(root="/home/antonio/PycharmProjects/ViT-MLP/Sydney/train",
                                   transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
    model = MLP(num_labels=7)

    epochs = 10
    save_path = r'/home/antonio/PycharmProjects/ViT-MLP/huggingface-vit-finetune/lightning_logs/version_0'

    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/antonio/PycharmProjects/ViT-MLP/huggingface-vit-finetune/lightning_logs/version_0",
        filename="best_model",
        save_top_k=1,
        mode="min",
    )

    pl.seed_everything(42)
    trainer = pl.Trainer(auto_scale_batch_size='power', gpus=1, deterministic=True, max_epochs=epochs,
                         auto_lr_find=True, benchmark=True, callbacks=[checkpoint_callback])
    if not os.path.exists(save_path):
        trainer.fit(model, train_loader)

    model = model.load_from_checkpoint(save_path + "/best_model-v1.ckpt", num_labels=7)
    trainer.test(model, test_loader)