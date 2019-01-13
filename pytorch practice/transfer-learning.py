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
            transforms.RandomResizedCrop(224, (0.80, 1.0)),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # FIX these to actual values from data
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # FIX these to actual mean/std/etc values from data
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

#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
learning_rate = 1e-4
optimizer_conv = torch.optim.Adam(model_conv.fc.parameters(), lr=learning_rate)


# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    # These should be calculated from the data, here they were just copy-pasted from an example
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

inputs, classes = next(iter(dataloaders['train']))
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

def train_model_custom(net, criterion, optimizer, dataloaders, num_epochs=25):
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


model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, 100)
#model_conv = train_model_custom(model_conv, criterion, optimizer_conv, dataloaders, 3)

visualize_model(model_conv)

print('done')
