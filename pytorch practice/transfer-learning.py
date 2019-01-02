import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
import cv2


def import_party():
    img_path = './cards/partypoker/training/'
    files = glob.glob('%s/*.png' % img_path)
    imgs = []
    for path in files:
        card_label = path.split('\\')[1][0:-4]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(([img], [card_label]))
    return imgs

data = import_party()

plt.ion()   # interactive mode


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model_conv = torchvision.models.resnet18(pretrained=True)
#for param in model_conv.parameters():
for param_name, param in model_conv.named_parameters():
    print("param name %s, %s" % (param_name, param.shape))
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features

# 52 different cards
model_conv.fc = nn.Linear(num_ftrs, 52) 

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

def train_model(net, criterion, optimizer, data):
    for epoch in range(2):
        running_loss = 0.0
        for i in range(len(data)):
            images, labels = data[i]
            images = torch.tensor(images)
            labels = torch.tensor(labels)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

model_conv = train_model(model_conv, criterion, optimizer_conv, data)
