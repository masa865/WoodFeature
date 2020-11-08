import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#function that creates a dataset labeled with the folder name under root_path
def make_dataset(root_path):
    #create Path object for root folder
    data_root = pathlib.Path(root_path)

    #get label names
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    #create list of data path and shuffle it
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    #assign an index to label
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    #labeling data path
    all_image_labels = []
    for path in all_image_paths:
        label = label_to_index[pathlib.Path(path).parent.name]
        all_image_labels.append(label)

        for augment in range(7):
            all_image_labels.append(label)

    np_labels = np.array(all_image_labels) #convert to numpy array

    #make image dataset 
    imgs = []
    for filename in all_image_paths:
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        imgs.append(img_v)

        #回転で水増し
        imgs.append(cv2.rotate(img_v,cv2.ROTATE_90_CLOCKWISE))
        imgs.append(cv2.rotate(img_v,cv2.ROTATE_180))
        imgs.append(cv2.rotate(img_v,cv2.ROTATE_90_COUNTERCLOCKWISE))
        #反転して回転
        rev_img = cv2.flip(img_v,1) #左右反転
        imgs.append(rev_img)
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_CLOCKWISE))
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_180))
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_COUNTERCLOCKWISE))

        np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)

#test net
def testNet(imsize=64,channel=1):

    inputs = keras.layers.Input(shape=(imsize,imsize,channel))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs,outputs=out)

    return model

#return lr
def lr_schedul(epoch):
    x = 0.001
    if epoch >= 80:
        x = 0.001/4
    if epoch >= 150:
        x = 0.001/16
    return x


#test code for this module
if __name__ == '__main__':

    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    
    #import data
    (load_images,load_labels) = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000_cleansing')

    #divide into training data and test data(90%:10%)
    test_images = load_images[:int(len(load_images)*0.1)]
    train_images = load_images[int(len(load_images)*0.1):]
    test_labels = load_labels[:int(len(load_labels)*0.1)]
    train_labels = load_labels[int(len(load_labels)*0.1):]

    #normalization
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #reshape
    train_images = train_images.reshape(-1,64,64,1)
    test_images = test_images.reshape(-1,64,64,1)

    #setting model
    model = testNet(64,1)

    model.summary()

    lr_decay = tf.keras.callbacks.LearningRateScheduler(
    lr_schedul,
    verbose=1,)

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=200,
                    verbose=1,
                    validation_data=(test_images, test_labels),
                    callbacks=[lr_decay],
                    )

    #plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    #plot training & validation acces
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()