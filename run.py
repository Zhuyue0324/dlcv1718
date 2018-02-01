import sys
sys.version

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import math
import random
import os
from PIL import Image
from torch.utils.data import Dataset, sampler
from MyFolder import MyImageFolder
from visualisation import *
import matplotlib.pyplot as plt

# Parsing command-line

if len(sys.argv) == 6:
    WEIGHTS_PATH = sys.argv[1]
    IN_34 = '34' in WEIGHTS_PATH
    SLICE_WIDTH = int(sys.argv[2])
    PRINT_FREQUENCY = int(sys.argv[3])
    NB_EPOCHS = int(sys.argv[4])
else:
    print("Command should have been like:")
    print("\tpython3 run.py WEIGHTS_PATH SLICE_WIDTH PRINT_FREQUENCY NB_EPOCHS")
    print("but was: python3 "+str(sys.argv))
    print("will use default values instead.")
    WEIGHTS_PATH = "dlcv_weight34.pth"  # where the weights are saved in the end, for further reuse
    IS_34 = True
    SLICE_WIDTH = 32
    PRINT_FREQUENCY = 1024
    NB_EPOCHS = 1
# Data loading and pre-processing

LABELSCSV = pd.read_csv("labels.csv")
TRANSFORM = transforms.Compose(
     [transforms.ToTensor()])
TARGET_TRANSFORM = transforms.Compose(
     [transforms.ToTensor(),
     transforms.Normalize((0, 0, 0), (1/255, 1/255, 1/255))])
ROAD_LABELS = [1,2,7,8,9,10]#remove 8,9,10 to only keep the road
ROOT = '/mnt/disks/sdb1/cityscapes/'
LI8B = 'leftImg8bit/'
GT = 'gtFine/'
DATA_TRAIN = MyImageFolder(root1=ROOT+LI8B+'train', root2=ROOT+GT+'train_lido' , transform = TRANSFORM, target_transform= TARGET_TRANSFORM)
DATA_VAL = MyImageFolder(root1=ROOT+LI8B+'val', root2=ROOT+GT+'val_lido' , transform = TRANSFORM, target_transform= TARGET_TRANSFORM)
DATA_TEST = MyImageFolder(root1=ROOT+LI8B+'test', root2=ROOT+GT+'test_lido' , transform = TRANSFORM, target_transform= TARGET_TRANSFORM)

# setting up the StixelNet

if IS_34:
    from Mynet34 import PretrainedResNet34, MyNet
else:
    from Mynet50 import PretrainedResNet50, MyNet
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import ImageFile
import os.path

ImageFile.LOAD_TRUNCATED_IMAGES = True
NUM_CLASSES = 2
NB_TRAIN=len(DATA_TRAIN)
NB_VAL=len(DATA_VAL)
NB_TEST=len(DATA_TEST)
BATCH_SIZE = 1
USE_GPU = torch.cuda.is_available()

if IS_34:
    PRETRAINED_NET = PretrainedResNet34()
    PRETRAINED_NET.load_state_dict(models.resnet34(pretrained=True).state_dict())
else:
    PRETRAINED_NET = PretrainedResNet50()
    PRETRAINED_NET.load_state_dict(models.resnet50(pretrained=True).state_dict())

NET = MyNet(NUM_CLASSES, PRETRAINED_NET)
if USE_GPU:
    NET.cuda()

CRITERION = torch.nn.CrossEntropyLoss(ignore_index=-1)
if USE_GPU:
    CRITERION.cuda()
OPTIMIZER = optim.SGD(NET.parameters(), lr=0.01, momentum=0.9)

TRAIN_LOADER = DataLoader(DATA_TRAIN, batch_size=BATCH_SIZE, sampler=sampler.RandomSampler(DATA_TRAIN))
VAL_LOADER = DataLoader(DATA_VAL, batch_size=BATCH_SIZE)
TEST_LOADER = DataLoader(DATA_TEST, batch_size=BATCH_SIZE)

if os.path.exists(WEIGHTS_PATH):
    NET.load_state_dict(torch.load(WEIGHTS_PATH))
    print("Loaded weights at:"+WEIGHTS_PATH)
else:
    print("No pretrained weights found at:"+WEIGHTS_PATH)

# Training (and validation) steps

print("Start training")

NEEDS_NEW_EPOCH = True
LAST_TRAIN_LOSS = 10
LAST_VAL_LOSS = 10
# EPOCH=0
WIDTH = 2048
NUM_SLICES = WIDTH // SLICE_WIDTH  # added   "Integer Division"

for epoch in range(NB_EPOCHS):
    # while needNewEpoch:
    # --------------------------------------training period---------------------------------------
    running_loss = 0.0
    epochloss = 0.0
    numsample = 0

    NET.train()

    print("Slice width:  %.5d Number of Slices:  %.5d" % (WIDTH, NUM_SLICES))  # added

    for inputs, labels in TRAIN_LOADER:
        labels_masked = reduce(labels.numpy(), ROAD_LABELS)
        labels = torch.from_numpy(labels_masked)

        for i in range(NUM_SLICES):  ### added

            inputs_temp = inputs[:, :, :, i * SLICE_WIDTH: (i + 1) * SLICE_WIDTH]  ### added
            labels_temp = labels[:, :, i * SLICE_WIDTH: (i + 1) * SLICE_WIDTH]  ### added

            if USE_GPU:
                inputs_temp = inputs_temp.cuda()  ### changed
                labels_temp = labels_temp.cuda()  ### changed

            inputs_temp, labels_temp = Variable(inputs_temp), Variable(labels_temp)  ### changed
            # zero the parameter gradients
            OPTIMIZER.zero_grad()
            # forward + backward + optimize
            outputs = NET(inputs_temp)  ### changed
            loss = CRITERION(outputs, labels_temp)  ### changed
            loss.backward()
            OPTIMIZER.step()
            # print statistics
            running_loss += loss.data[0]
            epochloss += loss.data[0]
            numsample += BATCH_SIZE
            if numsample % PRINT_FREQUENCY == 0:  # printfrequence-1:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, numsample, running_loss / PRINT_FREQUENCY),
                      end='\r', flush=True)
                running_loss = 0.0

    newTrainLoss = epochloss / (NB_TRAIN * NUM_SLICES)  # changed
    print('The average loss of epoch ', epoch + 1, ' is ', newTrainLoss)
    torch.save(NET.state_dict(), WEIGHTS_PATH)
    # --------------------------------------validation period---------------------------------------
    meanCorrectProba = 0.0
    epochloss = 0.0
    numsample = 0
    NET.eval()
    print("Slice width:  %.5d Number of Slices:  %.5d" % (WIDTH, NUM_SLICES))  # added

    for inputs, labels in VAL_LOADER:
        labels_masked = reduce(labels.numpy(), ROAD_LABELS)
        labels = torch.from_numpy(labels_masked)

        for i in range(NUM_SLICES):  # added

            inputs_temp = inputs[:, :, :, i * SLICE_WIDTH: (i + 1) * SLICE_WIDTH]  ### added
            labels_temp = labels[:, :, i * SLICE_WIDTH: (i + 1) * SLICE_WIDTH]  ### added

            if USE_GPU:
                inputs_temp = inputs_temp.cuda()  # changed
                labels_temp = labels_temp.cuda()  # changed

            inputs_temp, labels_temp = Variable(inputs_temp), Variable(labels_temp)  # changed
            outputs = NET(inputs_temp)  # changed
            loss = CRITERION(outputs, labels_temp)  # changed
            meanProbability = np.exp(-loss.data[0])
            epochloss += loss.data[0]
            meanCorrectProba += meanProbability
            numsample += BATCH_SIZE

    newValLoss = epochloss / (NB_VAL * NUM_SLICES)
    print('The average validation loss is ', newValLoss)
    print('The average correctness of the validation data is ', meanCorrectProba / (NB_VAL * NUM_SLICES) * 100,
          '%')  # changed
    # --------------------------------------evaluate the necessity of a new epoch---------------------------------------
    if (LAST_VAL_LOSS - newValLoss < 0.01) and (LAST_TRAIN_LOSS - newTrainLoss < 0.01):
        needNewEpoch = False
    else:
        lastloss = newValLoss
        # epoch=epoch+1
    LAST_VAL_LOSS = newValLoss
    LAST_TRAIN_LOSS = newTrainLoss

print("End training")