import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

import os
cwd = os.getcwd()
print(cwd)

path_to_images = './cards/partypoker/'


#cross_l = (119, 50, 50)
#cross_d = (121, 56, 35)
#
#diamond_l = (190, 64, 70)
#diamond_d = (193, 68, 45)
#
#spade_l = (265, 8, 50)
#spade_d = (280, 4, 30)
#
#heart_l = (0, 65, 74)
#heart_d = (1, 73, 50)

# letters: A K Q J T
# suits: D H C S


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

def get_cards(path):
    card1_label = path.split('\\')[1].split()[0]
    card2_label = path.split('\\')[1].split()[1][0:-4]

    print(card1_label, card2_label)

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img[515:580, 565:740]
    #plt.imshow(img)
    #plt.show()
    

    card1 = img[:,:87]
    card2 = img[:,87:]
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

files = glob.glob('%s/*.jpg' % path_to_images)
files.extend(glob.glob('%s/*.png' % path_to_images))
cards = []
min_x = 10000
min_y = 10000
max_x = 0
max_y = 0
for file in files:
    print(file)
    card1, card2 = get_cards(file)
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
    card['pic'] = card['pic'][0:min_x, 0:min_y]
    print(card['pic'].shape)
    write_file(card['pic'], card['label'])

print(min_x, max_x, min_y, max_y)