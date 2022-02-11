import sys
import time

from matplotlib import pyplot as plt
from torchmetrics import Precision
from transformers import ViTFeatureExtractor
from torch.utils.data import DataLoader, ConcatDataset
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
from LeViT import LeViT
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

from img_model import ExtractFeatures

from scipy.ndimage import gaussian_filter

from CyTran import ConvTransformer

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

from data_loaders.data_loader import RemoteSensingDataset
from sklearn.model_selection import train_test_split

term_width = 10

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
  '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


if __name__ == '__main__':
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    num_labels = 21
    batch_size = 8
    num_workers = 8
    max_epochs = 100
    split = 0.8
    dataset_path = "UCM"
    full_path = "/home/antonio/PycharmProjects/ViT-MLP/" + dataset_path

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

    transform_train = transforms.Compose([
        RandomRotation([-5, 5]),
        RandomVerticalFlip(0.3),
        RandomHorizontalFlip(0.3),
        ViTFeatureExtractorTransforms(model_name_or_path, feature_extractor),
        ExtractFeatures(feature_extractor)
    ])
    transform_test = transforms.Compose([
        ViTFeatureExtractorTransforms(model_name_or_path, feature_extractor),
        ExtractFeatures(feature_extractor)
    ])

    train_dataset = RemoteSensingDataset(dataset_dir=full_path, type='train', split=split, transform=transform_train)
    test_dataset = RemoteSensingDataset(dataset_dir=full_path, type='test', split=split, transform=transform_test)
    dataset = ConcatDataset([train_dataset, test_dataset])

    k_folds = 4
    results = {}
    kfold = KFold(n_splits=k_folds, shuffle=True)

    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=8, sampler=train_subsampler, num_workers=8)
        test_loader = DataLoader(dataset, batch_size=1, sampler=test_subsampler, num_workers=8)

        model = ConvTransformer(input_nc=6, out_classes=num_labels, n_downsampling=3, depth=9, heads=6, dropout=0.5)
        model.apply(reset_weights)

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)

        # ---------------- CLASSIC TRAINING ----------------

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        save_path_classic = '/home/antonio/PycharmProjects/ViT-MLP/ViT-MLP/ViT+CT_net_' + dataset_path + '.pth'
        # model = MLP_classic(num_labels).to(device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        acc_list = []

        # if not os.path.exists(save_path_classic):
        save_path = f'./model-fold-{fold}.pth'

        model.train()
        for epoch in range(max_epochs):
            print(f'Starting epoch {epoch + 1}')

            current_loss = 0.0
            train_loss = 0
            reg_loss = 0
            correct = 0
            total = 0
            best_acc = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, targets = data

                if max_epochs - 50 > epoch > 0:
                    inputs = inputs.numpy()
                    for batch in range(inputs.shape[0]):
                        for channel in range(3):
                            temp = gaussian_filter(inputs[batch, channel, :, :], sigma=0.5 / (5 * epoch))
                    inputs = torch.from_numpy(inputs)

                inputs = inputs.to(device)
                targets = targets.to(device)

                # inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                # inputs, targets_a, targets_b = map(Variable, (inputs,
                #                                               targets_a, targets_b))
                # outputs = model(inputs)
                # loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
                # train_loss += loss.item()
                # _, predicted = torch.max(outputs.data, 1)
                # total += targets.size(0)
                # correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                #             + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, targets)

                loss.backward()
                optimizer.step()

                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
                    current_loss = 0.0

                # progress_bar(i, len(train_loader),
                #              'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                #              % (train_loss / (i + 1), reg_loss / (i + 1),
                #                 100. * correct / total, correct, total))

            correct = 0
            total = 0
            with torch.no_grad():
                model.eval()
                for data in tqdm(test_loader):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print('Accuracy of the network on the test images: %f %%' % (
                    100 * correct / total))
            acc_list.append(100 * correct / total)

            if (100 * correct / total) > best_acc:
                best_acc = 100 * correct / total
                # torch.save(model.state_dict(), save_path_classic)
                torch.save(model.state_dict(), save_path)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        # save_path = f'./model-fold-{fold}.pth'
        # torch.save(model.state_dict(), save_path)

        model.load_state_dict(torch.load(save_path))
        model.eval()
        model.to(device)

        correct = 0.0
        total = 0.0
        y_pred = []
        y_true = []

        precision = Precision(num_classes=num_labels, top_k=3)
        precision = precision.to(device)

        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                precision(outputs, labels)

                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %f %%' % (
                100 * correct / total))

        # Print accuracy
        print('Accuracy for fold %d: %f %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)

        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum / len(results.items())} %')

    torch.cuda.empty_cache()

    cf_matrix = confusion_matrix(y_true, y_pred)
    num_classes = list(set(y_true))

    cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cf_matrix,
                         index=[i for i in range(len(num_classes))],
                         columns=[i for i in range(len(num_classes))])
    df_cm = df_cm.round(1)

    sns.set(font_scale=1, rc={'text.usetex': True})
    # sns.set_context("paper", rc={"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 5})

    plt.figure(figsize=(12, 7))
    plot = sns.heatmap(df_cm, annot=True)
    # plot.set_xticks(range(len(num_classes)))
    # plot.set_xticklabels(
    #     ['0', ' ', ' ', '3', ' ', ' ', '6', ' ', ' ', '9', ' ', ' ', '12', ' ', ' ', '15', ' ', ' ', '18', ' ', ' ',
    #      '21', ' ', ' ', '24', ' ', ' ', '27', ' ', '29'])
    # plot.set_yticks(range(len(num_classes)))
    # plot.set_yticklabels(
    #     ['0', ' ', ' ', '3', ' ', ' ', '6', ' ', ' ', '9', ' ', ' ', '12', ' ', ' ', '15', ' ', ' ', '18', ' ', ' ',
    #      '21', ' ', ' ', '24', ' ', ' ', '27', ' ', '29'])
    # plot.set_xticklabels(
    #     ['0', ' ', ' ', '3', ' ', ' ', '6', ' ', ' ', '9', ' ', ' ', '12', ' ', ' ', '15', ' ', ' ', '18', ' ', '20'])
    # plot.set_yticks(range(len(num_classes)))
    # plot.set_yticklabels(
    #     ['0', ' ', ' ', '3', ' ', ' ', '6', ' ', ' ', '9', ' ', ' ', '12', ' ', ' ', '15', ' ', ' ', '18', ' ', '20'])
    plt.savefig(dataset_path + "_" + 'confusion_matrix.png')

    # Compute per-class accuracy
    print("Accuracy for class: ")
    for idx in num_classes:
        print(idx, end=" : ")
        print(cf_matrix.diagonal()[idx])

    print("Precision@K: ", precision.compute().cpu().numpy())
