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

suit_map = {
    'd': 0,
    'h': 1,
    'c': 2,
    's': 3,
}

anti_suit_map = {
    0: 'd',
    1: 'h',
    2: 'c',
    3: 's',
}

val_map = {
    'A': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 7,
    '9': 8,
    'T': 9,
    'J': 10,
    'Q': 11,
    'K': 12,
}

anti_val_map = {
    0: 'A',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: 'T',
    10: 'J',
    11: 'Q',
    12: 'K',
}

width = len(suit_map)

def get_idx_from_str(label):
    return get_idx(label[0], label[1])

def get_idx(val, suit):
    return width * val_map[val] + suit_map[suit]

def un_idx(idx):
    suit = int(idx / 13)
    val = idx % 13
    return (suit, val)

def import_loader():
    data_dir = './cards/torch/'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, (1.0, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # FIX these to actual values from data
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: torchvision.datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                                    for x in ['train', 'val']
    }
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0)
                                                for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return (image_datasets, dataloaders, dataset_sizes, class_names)



(image_datasets, dataloaders, dataset_sizes, class_names) = import_loader()

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

def train_model(net, criterion, optimizer, dataloaders, num_epochs=25):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        i = 0
        for inputs, labels in dataloaders['train']:
            #images, labels = data[i]
            #images = torch.tensor(images)
            #labels = torch.tensor(labels)
            images, labels = inputs.to(device), labels.to(device)

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
            i += 1
    return net


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

print('start train')

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

model_conv = train_model(model_conv, criterion, optimizer_conv, dataloaders, 25)

visualize_model(model_conv)

print('done')
