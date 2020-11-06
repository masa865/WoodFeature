import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random

import cv2

from sklearn.model_selection import GridSearchCV

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
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        imgs.append(img_v)

        np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)

#Model
def testNet(activation="sigmoid", optimizer="adam", out_dim=64):

    inputs = keras.layers.Input(shape=(imsize,imsize,channel))
    x = keras.layers.Conv2D(out_dim, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(out_dim, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(out_dim, activation='relu')(x)
    out = keras.layers.Dense(1, activation=activation)(x)
    model = keras.Model(inputs=inputs,outputs=out)

    model.compile(optimizer=optimizer,loss='binary_crossentropy',
              metrics=['accuracy'])

    return model

#test code for this module
if __name__ == '__main__':

    #parameter for grid search
    activation = ["sigmoid"]
    optimizer = ["adam"]
    out_dim = [16,32,64]
    nb_epoch = [100]
    batch_size = [32,64,128]

    test_model = testNet()
    model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=test_model,n_jobs=-1)

    param_grid = dict(activation=activation, 
                  optimizer=optimizer, 
                  out_dim=out_dim, 
                  nb_epoch=nb_epoch, 
                  batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    
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
    model = testNet(train_images)

    grid_result = grid.fit(train_images, train_labels)
    print (grid_result.best_score_)
    print (grid_result.best_params_)

    
