import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed = 1

def load_card_files(card_path):
    files = glob.glob(card_path)
    x = []
    y = []
    for path in files:
        # works only on windows for now
        # take last section in path after \, cut out .png at the end
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

def preview():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
    datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')

    # hard code some random image from preprocessed cards
    img = load_img('./cards/torch/val/8h/9a31d523-0f59-4e8b-8c97-d872247ecd6b.png')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # Ugly looping because we want to consume the units
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
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
                    print('%s%s correctly identified' % (val1, suit1))
                else:
                    print('%s%s confused with %s%s' % (val1, suit1, val2, suit2))

    print('suit fails %s val fails %s both fails %s' % (suit_fails, val_fails, both_failed))

def print_model_stats(model, val_gen):
    evaluation = model.evaluate_generator(val_gen)
    # Check confusions...
    #predictions = model.predict_generator(val_gen).argmax(axis=-1)
    predictions = model.predict_generator(val_gen).argmax(axis=-1)
    #predictions = model.predict_generator(val_gen)
    #predictions = predictions.argmax(axis=1)
    
    #y_validate = predictions.argmax(axis=-1)
    #predictions = model.predict_classes(x_validate)
    confusion = tf.confusion_matrix(val_gen.classes, predictions).eval()
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


def plot_history(acc, val_acc, loss, val_loss):
    plt.subplot(2, 1, 1)

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['acc', 'val acc'], loc='upper left')


    plt.subplot(2, 1, 2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['loss', 'val loss'], loc='upper left')
    plt.show()

#mnist_model = mnist_for_cards()
#print_model_stats(mnist_model)

def conv_cards(train_gen, val_gen, card_size, batch_size, epochs):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=card_size, activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(256, (5, 5), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Conv2D(64, (20, 20), padding="same", activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.10),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.15),


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(52 * 50, activation=tf.nn.relu), 
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(train_gen.num_classes, activation=tf.nn.softmax)
    ])
    #optimizer = tf.keras.optimizers.SGD(lr=0.01, nesterov=True)
    optimizer = tf.keras.optimizers.Adam()
    # optimizer = tf.keras.optimizers.SGD(lr=0.1, momentum=0.5, decay=0.0, nesterov=False)

    model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                #loss='categorical_crossentropy',
                metrics=['accuracy'])

    steps_per_epoch = train_gen.n // batch_size
    validation_steps = val_gen.n // batch_size
    if validation_steps < 1:
        validation_steps = 1
    model.fit_generator(
        train_gen,
        #steps_per_epoch=train_gen.n,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        ##validation_steps=val_gen.n,
        validation_steps=validation_steps,
        #batch_size=batch_size
        shuffle=True,
        #shuffle=False,
        #use_multiprocessing=True,
    )
    return model

def load_training(card_size, batch_size):
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            #width_shift_range=0.2,
            #height_shift_range=0.1,
            #zoom_range=0.2,
            #fill_mode='nearest'
        )
    gen = train_datagen.flow_from_directory(
        './cards/torch/train', # mebbe rename since using from tf as well kjeh
        target_size=card_size,
        batch_size=batch_size,
        #save_to_dir='preview',
        class_mode="sparse",
        shuffle=True,
        #shuffle=False,
        seed=seed
    )

    show_imgs = False
    if show_imgs == True:
        for batch in gen:
            img = batch[0][0]
            plt.imshow(img)
            plt.show()
            print("ses")
    return gen


def load_validation(card_size, batch_size):
    load_datagen = ImageDataGenerator(
        rescale=1./255
    )
    return load_datagen.flow_from_directory(
        './cards/torch/val', # mebbe rename 'torch' since using from tf as well
        target_size=card_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
        seed=seed
    )

# draw example
#preview()

print(tf.keras.backend.image_data_format())


card_size = (75, 90)
#card_size = (64, 75)
#card_size = (150, 150)
batch_size = 32 
epochs = 10
#x_train, y_train = load_training(card_size)
#x_validate, y_validate = load_validation(card_size)
train_gen = load_training(card_size, batch_size)
val_gen = load_validation(card_size, batch_size)

sess = tf.InteractiveSession()

in_shape = (card_size[0], card_size[1], 3)
cnn_model = conv_cards(train_gen, val_gen, in_shape, batch_size, epochs)
print_model_stats(cnn_model, val_gen)

train_acc = cnn_model.history.history['acc']
test_acc = cnn_model.history.history['val_acc']

loss = cnn_model.history.history['loss']
val_loss = cnn_model.history.history['val_loss']

plot_history(train_acc, test_acc, loss, val_loss)