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
    np_labels = np.array(all_image_labels) #convert to numpy array

    #make image dataset 
    imgs = []
    for filename in all_image_paths:
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        imgs.append(img_v)

        np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)

#All Convolutional Net
def allConvNet(imsize=64,channel=1):

    inputs = keras.layers.Input(shape=(imsize,imsize,channel))
    x = keras.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs,outputs=out)
    print("allConvNet:")
    model.summary()

    return model

#baseline net
def baseLineNet(inputs,imsize=64,channel=1):

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

#test net
def testNet(imsize=64,channel=1):

    inputs = keras.layers.Input(shape=(imsize,imsize,channel))
    x = keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    out = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs,outputs=out)

    return model

#test code for this module
if __name__ == '__main__':

    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split

    lr=0.0004
    batch_size = 128
    epochs = 1500
    optimizer = "adamax"
    
    #import data
    (train_images,train_labels) = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000_cleansing')

    #divide into training data and test data(90%:10%)
    #test_images = load_images[:int(len(load_images)*0.1)]
    #train_images = load_images[int(len(load_images)*0.1):]
    #test_labels = load_labels[:int(len(load_labels)*0.1)]
    #train_labels = load_labels[int(len(load_labels)*0.1):]

    #normalization
    train_images = train_images / 255.0
    #test_images = test_images / 255.0

    #reshape
    train_images = train_images.reshape(-1,64,64,1)
    #test_images = test_images.reshape(-1,64,64,1)

    #setting model
    #base_model = baseLineNet(train_images)
    model = testNet(64,1)

    model.summary()

    model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

    # Fit the model
    history = model.fit(train_images, train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1)
    

    #plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss(binary cross enrropy)')
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

    #save model
    #model.save('base_model.h5')
    model.save('test_model.h5')
