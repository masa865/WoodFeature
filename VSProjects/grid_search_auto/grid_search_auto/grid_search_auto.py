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
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

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

        clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(3,3))
        img_v = clahe.apply(img_v)

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
def gridSearch(train_data,train_label,test_data,test_label,
              activation,optimizer,epochs,batch_size,learn_rate,out_dim1,out_dim2,out_dim3,out_dim4): #parameter lists

    if not (os.path.exists('result')):
        os.mkdir('result')
    path = 'result/result.csv'
    if os.path.exists(path):
        os.remove(path)

    with open(path,'a',newline='') as c:
        writer = csv.writer(c)
        writer.writerow(['activation','optimizer','epochs','batch_size','learn_rate','out_dim1','out_dim2','out_dim3','out_dim4',
                         'CV_ACC','CV_STD','Test_ACC','Test_STD'])

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # define X-fold cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

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
                                        test_acces = []
                                        for train, test in kfold.split(X,Y):
                                            # create model
                                            model = getModel(ac,ou1,ou2,ou3,ou4)

                                            model.compile(optimizer=op,
                                                    loss='binary_crossentropy',
                                                    metrics=['accuracy'])
                                            tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

                                            # Fit the model
                                            history = model.fit(X[train], Y[train],
                                                            batch_size=ba,
                                                            epochs=ep,
                                                            validation_data=(X[test],Y[test]),
                                                            verbose=1)

                                            # Evaluate
                                            scores = model.evaluate(X[test], Y[test], verbose=0)
                                            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                                            cvscores.append(scores[1] * 100)

                                            test_loss, test_acc = model.evaluate(test_data, test_label)
                                            test_acces.append(test_acc*100)

                                        mean = np.mean(cvscores)
                                        std = np.std(cvscores)
                                        print("%.2f%% (+/- %.2f%%)" % (mean, std))

                                        test_mean = np.mean(test_acces)
                                        test_std = np.std(test_acces)

                                        #plot training & validation loss values
                                        plt.plot(history.history['loss'])
                                        plt.plot(history.history['val_loss'])
                                        plt.title('Model loss')
                                        plt.ylabel('Loss')
                                        plt.xlabel('Epoch')
                                        plt.legend(['Train', 'Validation'], loc='upper left')
                                        plt.show()

                                        #plot training & validation acces
                                        plt.plot(history.history['accuracy'])
                                        plt.plot(history.history['val_accuracy'])
                                        plt.title('Model accuracy')
                                        plt.ylabel('Accuracy')
                                        plt.xlabel('Epoch')
                                        plt.legend(['Train', 'Validation'], loc='upper left')
                                        plt.show()

                                        #plot roc curve
                                        y_pred_keras = model.predict(test_data).ravel()
                                        fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_label, y_pred_keras)
                                        auc_keras = auc(fpr_keras, tpr_keras)
                                        plt.figure(1)
                                        plt.plot([0, 1], [0, 1], 'k--')
                                        plt.plot(fpr_keras, tpr_keras, label='Test data (area = {:.3f})'.format(auc_keras))
                                        plt.xlabel('False positive rate')
                                        plt.ylabel('True positive rate')
                                        plt.title('ROC curve')
                                        plt.legend(loc='best')
                                        plt.show()

                                        #write result
                                        with open(path,'a',newline='') as c:
                                                writer = csv.writer(c)
                                                writer.writerow([ac,op,ep,ba,lr,ou1,ou2,ou3,ou4,
                                                                mean,std,test_mean,test_std])

    return model

#test code for this module
if __name__ == '__main__':

    #parameter list for grid search
    activation = ["sigmoid"]
    optimizer = ["adamax"]
    epochs = [1500]
    batch_size = [128]
    learn_rate = [0.0004]
    out_dim1 = [16]
    out_dim2 = [32]
    out_dim3 = [32]
    out_dim4 = [16]

    #import data
    (train_images,train_labels) = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000_cleansing')
    (test_images,test_labels) = make_dataset(r'E:\traning_data(murakami)\test_data_50012')

    #divide into training data and test data(90%:10%)
    #test_images = load_images[:int(len(load_images)*0.1)]
    #train_images = load_images[int(len(load_images)*0.1):]
    #test_labels = load_labels[:int(len(load_labels)*0.1)]
    #train_labels = load_labels[int(len(load_labels)*0.1):]

    #normalization
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #reshape
    train_images = train_images.reshape(-1,64,64,1)
    test_images = test_images.reshape(-1,64,64,1)

    #learn
    model = gridSearch(train_images,train_labels,test_images,test_labels,
                       activation,optimizer,epochs,batch_size,learn_rate,out_dim1,out_dim2,out_dim3,out_dim4)

    model.save('model_64_10fold.h5')

    
