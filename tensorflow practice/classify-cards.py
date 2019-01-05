import cv2
import tensorflow as tf
import numpy as np
#mnist = tf.keras.datasets.mnist
import glob


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

def get_idx(val, suit):
    return width * val_map[val] + suit_map[suit]

def un_idx(idx):
    suit = int(idx / 13)
    val = idx % 13
    return (suit, val)

def load_card_files(card_path):
    files = glob.glob(card_path)
    x = []
    y = []
    for path in files:
        # take last section after \, cut out .png at the end
        card_label = path.split('\\')[1].split()[0][:-4] 
        val = card_label[0]
        suit = card_label[1]
        idx = get_idx(val, suit)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x.append(img / 255)
        y.append(idx)

        #print(card_label)
        #print(idx)
    return np.array(x), np.array(y)

def load_training():
    return load_card_files('./cards/winner-poker/training/*.png')

def load_validation():
    return load_card_files('./cards/winner-poker/validation/*.png')

x_train, y_train = load_training()
x_validate, y_validate = load_validation()

sess = tf.InteractiveSession()

def print_confusion(confusion):
    suit_fails = 0
    val_fails = 0
    both_failed = 0
    for row_idx in range(len(confusion)):
        for col_idx in range(len(confusion[row_idx])):
            if confusion[row_idx][col_idx] != 0:
                (suit1, val1) = un_idx(row_idx)
                (suit2, val2) = un_idx(col_idx)
                suit1 = anti_suit_map[suit1]
                suit2 = anti_suit_map[suit2]
                val1 = anti_val_map[val1]
                val2 = anti_val_map[val2]

                if val1 != val2:
                    val_fails += 1

                if suit1 != suit2:
                    suit_fails += 1

                if suit1 != suit2 and val1 != val2:
                    both_failed += 1

                if row_idx == col_idx:
                    print('%s%s correctly identified as %s%s' % (val1, suit1, val2, suit2))
                else:
                    print('%s%s confused with %s%s' % (val1, suit1, val2, suit2))

    print('suit fails %s val fails %s both fails %s' % (suit_fails, val_fails, both_failed))

def print_model_stats(model):
    evaluation = model.evaluate(x_validate, y_validate)
    # Check confusions...
    predictions = model.predict_classes(x_validate)
    confusion = tf.confusion_matrix(y_validate, predictions).eval()
    print_confusion(confusion)
    print('validation accuracy', evaluation[1])

def mnist_for_cards():
    # almost directly using MNIST example, around 50-66% accuracy
    # often hits almost exactly 66% for some reason. There are 3 color channels...?
    # No directly obvious patterns in missclassification
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        # >= 2 multiples of the number of different cards seem to work well
        tf.keras.layers.Dense(52 * 2, activation=tf.nn.relu), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(52, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=120)
    return model


#mnist_model = mnist_for_cards()
#print_model_stats(mnist_model)

def conv_cards(dada):
    in_shape = dada[0].shape
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=in_shape, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        #tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Conv2D(32, (5, 5), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),

        tf.keras.layers.Flatten(),
        #tf.keras.layers.Flatten(input_shape=card_size),
        tf.keras.layers.Dense(52 * 20, activation=tf.nn.relu), 
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(52, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=200, batch_size=10)
    return model


cnn_model = conv_cards(x_train)
print_model_stats(cnn_model)