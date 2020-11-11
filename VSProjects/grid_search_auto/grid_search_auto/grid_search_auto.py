import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import os
import csv
import sys

from sklearn.model_selection import StratifiedKFold

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
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        imgs.append(img_v)

        np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)

def getModel(ac,ou1,ou2,ou3,ou4):
    inputs = keras.layers.Input(shape=(64,64,1))
    x = keras.layers.Conv2D(ou1, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(ou2, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(ou3, (3, 3), activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(ou4, activation='relu')(x)
    out = keras.layers.Dense(1, activation=ac)(x)
    model = keras.Model(inputs=inputs,outputs=out)

    return model

#Model
def gridSearch(train_data,train_label,
              activation,optimizer,epochs,batch_size,learn_rate,out_dim1,out_dim2,out_dim3,out_dim4): #parameter lists

    if not (os.path.exists('result')):
        os.mkdir('result')
    path = 'result/result.csv'
    if os.path.exists(path):
        os.remove(path)

    with open(path,'a',newline='') as c:
        writer = csv.writer(c)
        writer.writerow(['activation','optimizer','epochs','batch_size','learn_rate','out_dim1','out_dim2','out_dim3','out_dim4',
                         'CV_ACC','CV_STD'])

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # define X-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    X=train_data
    Y=train_label

    for ac in activation:
        for op in optimizer:
            for ep in epochs:
                for ba in batch_size:
                    for lr in learn_rate:
                        for ou1 in out_dim1:
                            for ou2 in out_dim2:
                                for ou3 in out_dim3:
                                    for ou4 in out_dim4:
                                        #cross validation
                                        cvscores = []
                                        for train, test in kfold.split(X,Y):
                                            # create model
                                            model = getModel(ac,ou1,ou2,ou3,ou4)

                                            model.compile(optimizer=op,
                                                    loss='binary_crossentropy',
                                                    metrics=['accuracy'])
                                            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

                                            # Fit the model
                                            model.fit(X[train], Y[train],
                                                            batch_size=ba,
                                                            epochs=ep,
                                                            verbose=1)

                                            # Evaluate
                                            scores = model.evaluate(X[test], Y[test], verbose=0)
                                            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                                            cvscores.append(scores[1] * 100)

                                        mean = np.mean(cvscores)
                                        std = np.std(cvscores)
                                        print("%.2f%% (+/- %.2f%%)" % (mean, std))

                                        #write result
                                        with open(path,'a',newline='') as c:
                                                writer = csv.writer(c)
                                                writer.writerow([ac,op,ep,ba,lr,ou1,ou2,ou3,ou4,
                                                                mean,std])

    return model

#test code for this module
if __name__ == '__main__':

    #parameter list for grid search
    activation = ["sigmoid"]
    optimizer = ["adamax"]
    epochs = [800]
    batch_size = [128]
    learn_rate = [0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001]
    out_dim1 = [16]
    out_dim2 = [32]
    out_dim3 = [32]
    out_dim4 = [16]

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

    #learn
    model = gridSearch(train_images,train_labels,
                       activation,optimizer,epochs,batch_size,learn_rate,out_dim1,out_dim2,out_dim3,out_dim4)

    
