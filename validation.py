import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torchvision import transforms
import argparse
import tensorboardX
import os
import random
import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import AverageMeter, calculate_accuracy

# def calculate_all_matrics(outputs,targets):
#
#     batch_size = targets.size(0)
#
#     _, pred = outputs.topk(1, 1, True)
#     pred = pred.t()
#     correct = pred.eq(targets.view(1, -1))
#     n_correct_elems = correct.float().sum().item()
#
#     # return acc,precision,recall,f1

def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    # precisions = AverageMeter()
    # recalls = AverageMeter()
    # f1s = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  

            loss = criterion(outputs, targets)
            acc= calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))
            # accuracies.update(pre, data.size(0))
            # recalls.update(rec, data.size(0))
            # f1s.update(f1, data.size(0))

    # show info
    print('Validation set ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg, accuracies.avg * 100))
    return losses.avg, accuracies.avg

    