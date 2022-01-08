from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, RandomVerticalFlip, RandomHorizontalFlip, \
    RandomResizedCrop, RandomRotation

import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl

import os
from tqdm import tqdm
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint

from ViT import ViT
from vit_pytorch.levit import LeViT
from sklearn.model_selection import GridSearchCV
import numpy as np

from img_model import ExtractFeatures

from CyTran import ConvTransformer

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
    def __init__(self, model_name_or_path, feature_extractor):
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
    num_labels = 21
    batch_size = 8
    num_workers = 2
    max_epochs = 50

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    transform_train = transforms.Compose([
        RandomResizedCrop(224),
        RandomRotation([-45, 45]),
        RandomVerticalFlip(0.3),
        RandomHorizontalFlip(0.3),
        ViTFeatureExtractorTransforms(model_name_or_path, feature_extractor),
        ExtractFeatures(feature_extractor)
    ])
    transform_test = transforms.Compose([
        ViTFeatureExtractorTransforms(model_name_or_path, feature_extractor),
        ExtractFeatures(feature_extractor)
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

    # ---------------- PYTORCH-LIGHTNING TRAINING ----------------

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="/home/antonio/PycharmProjects/ViT-MLP/ViT-MLP/version_0",
    #     filename="best_model",
    #     save_top_k=1,
    #     mode="min",
    # )

    # save_path_pl = r'/home/antonio/PycharmProjects/ViT-MLP/ViT-MLP/version_0'
    # model = MLP_pl(num_labels)
    #
    # pl.seed_everything(42)
    # trainer = pl.Trainer(auto_scale_batch_size='power', gpus=1, deterministic=True, max_epochs=max_epochs,
    #                      auto_lr_find=True, benchmark=True, callbacks=[checkpoint_callback])
    # if not os.path.exists(save_path_pl):
    #     trainer.fit(model, train_loader)
    #
    # model = model.load_from_checkpoint(save_path_pl + "/best_model-v1.ckpt", num_labels=num_labels)
    # trainer.test(model, test_loader)

    # model = ViT(
    #     image_size=224,
    #     patch_size=32,
    #     num_classes=num_labels,
    #     dim=512,
    #     depth=3,
    #     heads=6,
    #     mlp_dim=512,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )

    model = LeViT(
        image_size=224,
        num_classes=num_labels,
        stages=1,  # number of stages
        dim=(256),  # dimensions at each stage
        depth=4,  # transformer of depth 4 at each stage
        heads=(6),  # heads at each stage
        mlp_mult=2,
        dropout=0.1
    )

    # model = ConvTransformer()

    # ---------------- CLASSIC TRAINING ----------------

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    save_path_classic = '/home/antonio/PycharmProjects/ViT-MLP/ViT-MLP/ViT+MLP_net.pth'
    # model = MLP_classic(num_labels).to(device)
    torch.manual_seed(42)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if not os.path.exists(save_path_classic):
        for epoch in range(max_epochs):
            print(f'Starting epoch {epoch + 1}')

            current_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, targets)

                loss.backward()
                optimizer.step()

                current_loss += loss.item()
                if i % 20 == 19:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 20))
                    current_loss = 0.0
        # torch.save(model.state_dict(), save_path_classic)

    print('Training process has finished.')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    torch.cuda.empty_cache()
