import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#All Convolutional Net
def allConvNet(inputs,width=28,height=28,channel=1):

    inputs = keras.layers.Input(shape=(28,28,1))
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

#Conv and Pool CNN Net
def convPoolNet(inputs,width=28,height=28,channel=1):

    inputs = keras.layers.Input(shape=(28,28,1))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    out = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs,outputs=out)
    print("convPoolNet:")
    model.summary()

    return model

#test code for this module
if __name__ == '__main__':
    
    #use mnist
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    #normalization
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #reshape
    train_images = train_images.reshape(-1,28,28,1)
    test_images = test_images.reshape(-1,28,28,1)

    model = convPoolNet(train_images)
    model = allConvNet(train_images)
