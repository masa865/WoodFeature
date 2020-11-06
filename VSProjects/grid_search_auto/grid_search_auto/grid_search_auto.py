import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import os
import csv

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

#Model
def gridSearch(train_data,train_label,val_data,val_label,
              activation,optimizer,out_dim,epoch,batch_size): #parameter lists

    acc_target = 0.95

    if not (os.path.exists('result')):
        os.mkdir('result')
    path = 'result/result.csv'
    if os.path.exists(path):
        os.remove(path)

    with open(path,'a',newline='') as c:
        writer = csv.writer(c)
        writer.writerow(['activation','optimizer','out_dim','epoch','batch_size','maxValAcc'])


    for ac in activation:
        for op in optimizer:
            for ou in out_dim:
                for ep in epoch:
                    for ba in batch_size:
                        #--Model definition part----------------------------------------------
                        inputs = keras.layers.Input(shape=(64,64,1))
                        x = keras.layers.Conv2D(ou, (3, 3), activation='relu')(inputs)
                        x = keras.layers.MaxPooling2D((2, 2))(x)
                        x = keras.layers.Conv2D(ou, (3, 3), activation='relu')(x)
                        x = keras.layers.MaxPooling2D((2, 2))(x)
                        x = keras.layers.Conv2D(ou, (3, 3), activation='relu')(x)
                        x = keras.layers.Flatten()(x)
                        x = keras.layers.Dense(ou, activation='relu')(x)
                        out = keras.layers.Dense(1, activation=ac)(x)
                        model = keras.Model(inputs=inputs,outputs=out)
                        #---------------------------------------------------------------------

                        #compile
                        model.compile(optimizer=op,
                            loss='binary_crossentropy',
                            metrics=['accuracy'])

                        #fit
                        history=model.fit(train_data,
                            train_label,
                            epochs=ep,
                            batch_size=ba,
                            validation_data=(val_data,val_label))

                        #save training & validation loss values
                        plt.figure()
                        plt.plot(history.history['loss'])
                        plt.plot(history.history['val_loss'])
                        plt.title('Model loss')
                        plt.ylabel('Loss')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Val'], loc='upper left')
                        plt.savefig('result/Loss_{0}_{1}_{2}_{3}_{4}.png'.format(ac,op,ou,ep,ba))

                        #save training & validation acces
                        plt.figure()
                        plt.plot(history.history['accuracy'])
                        plt.plot(history.history['val_accuracy'])
                        plt.hlines(acc_target,0,ep, "blue", linestyles='dashed')
                        plt.title('Model accuracy')
                        plt.ylabel('accuracy')
                        plt.xlabel('Epoch')
                        plt.legend(['Train', 'Val'], loc='upper left')
                        plt.savefig('result/Acc_{0}_{1}_{2}_{3}_{4}.png'.format(ac,op,ou,ep,ba))

                        #write result
                        with open(path,'a',newline='') as c:
                             writer = csv.writer(c)
                             writer.writerow([ac,op,ou,ep,ba,max(history.history['val_accuracy'])])

    return model

#test code for this module
if __name__ == '__main__':

    #parameter for grid search
    activation = ["sigmoid"]
    optimizer = ["adam"]
    out_dim = [16,32]
    nb_epoch = [10,20]
    batch_size = [32]
    
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

    #learn
    model = gridSearch(train_images,train_labels,test_images,test_labels,
                       activation,optimizer,out_dim,nb_epoch,batch_size)

    
