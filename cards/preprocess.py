import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import uuid
import random

import os
cwd = os.getcwd()
print(cwd)

# (x|y)min / (x|y)max give card positions in the default size table
paths_to_images = [
    #{'path': './cards/winner-poker', 'ymin': 480, 'ymax': 555,
    #     'xmin': 490, 'xmax': 620, 'card_size': 64},
    #{'path': './cards/partypoker',   'ymin': 515, 'ymax': 580,
    #     'xmin': 565, 'xmax': 740, 'card_size': 87},
    {'path': './cards/888poker',   'ymin': 500, 'ymax': 580,
         'xmin': 520, 'xmax': 645, 'card_size': 65},
]

path_to_torch_images = './cards/torch/'


# Generates train set augmentations. 
# Augmenting could be done at training time with the generators,
#  but it's nice to get to look at the augmented pics.
def generate_mods(img, path):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    datagen = ImageDataGenerator(
            #rescale=1./255, # do rescaling at training time
            width_shift_range=0.2,
            height_shift_range=0.1,
            zoom_range=0.1,
            rotation_range=5,
            fill_mode='nearest')

    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Gotta loop it stupidly like this to consume the batches
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=path, save_prefix='cat', save_format='png'):
        i += 1
        if i > 30:
            break
        

# This uses augmented images as a train set, and the original images as the validation set
# The actual card images will always be in the same size and orientation anyway.
def write_file_torch(image, label):
    unique_filename = str(uuid.uuid4())

    path = '%s/val/%s' % (path_to_torch_images, label)
    os.makedirs(path, exist_ok=True)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('%s/%s.png' % (path, unique_filename), rgb_img)

    path = '%s/train/%s' % (path_to_torch_images, label)
    os.makedirs(path, exist_ok=True)
    generate_mods(image, path)


# Use this to split every fourth image to validate set
# def write_file_torch(image, label):
#     unique_filename = str(uuid.uuid4())
#     path = ''
#     # write about every tenth file to validation set
#     if random.randint(1, 4) == 4:
#         path = '%s/val/%s' % (path_to_torch_images, label)
#     else:
#         path = '%s/train/%s' % (path_to_torch_images, label)
#
#     os.makedirs(path, exist_ok=True)
#     cv2.imwrite('%s/%s.png' % (path, unique_filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

used_card_labels = []
def write_file(image, label):
    #print(image.shape)
    if label not in used_card_labels:
        used_card_labels.append(label)
        filename = '%s/training/%s.png' % (path_to_images, label)
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        filename = '%s/validation/%s.png' % (path_to_images, label)
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def compare(one, two):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(one['pic'])
    axarr[0].set_title(one['label'])
    axarr[1].imshow(two['pic'])
    axarr[1].set_title(two['label'])
    plt.show()

def get_cards(path, site):
    cards = path.split('\\')[1].split()[0][0:-4]
    card1_label = cards[0:2]
    card2_label = cards[2:4]

    print(card1_label, card2_label)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ymin = site['ymin']
    ymax = site['ymax']
    xmin = site['xmin']
    xmax = site['xmax']
    card_size = site['card_size']

    img = img[ymin:ymax, xmin:xmax]
    card1 = img[:, :card_size]
    card2 = img[:, card_size:]
    #plt.imshow(img)
    #plt.show()

    card1 = {
        'label': card1_label,
        'pic': card1
    }
    
    card2 = {
        'label': card2_label,
        'pic': card2
    }
    #compare(card1, card2) # draw cards next to each other with labels
    return (card1, card2)

for site in paths_to_images:
    path_to_images = site['path']

    files = glob.glob('%s/*.jpg' % path_to_images)
    files.extend(glob.glob('%s/*.png' % path_to_images))
    cards = []
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    for file in files:
        print(file)
        card1, card2 = get_cards(file, site)
        c = [card1, card2]

        for card in c:
            image = card['pic']
            if image.shape[0] < min_x:
                min_x = image.shape[0]
            if image.shape[1] < min_y:
                min_y = image.shape[1]

            if image.shape[0] > max_x:
                max_x = image.shape[0]
            if image.shape[1] > max_y:
                max_y = image.shape[1]

            cards.append(card)

    for card in cards:
        card['pic'] = card['pic'][0:min_x, 0:min_y] # make all the same size
        print(card['pic'].shape)
        write_file_torch(card['pic'], card['label'])

    print(min_x, max_x, min_y, max_y)
