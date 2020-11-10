#cross validation
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pathlib
import random

import cv2

def getModel():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    return model

def crossVal(train_images,train_labels,fold_num=5,batch_size=32,epochs=50):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # define X-fold cross validation
    kfold = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    cvscores = []

    X=train_images
    Y=train_labels

    for train, test in kfold.split(X,Y):
        # create model
        model = getModel()

        model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

        # Fit the model
        model.fit(X[train], Y[train],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)


        # Evaluate
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
    return np.mean(cvscores),np.std(cvscores)


#-------------test code for this module--------------------------------------------
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    from sklearn.datasets import make_classification
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc

    #root_path下のフォルダ名をラベルとするデータセットを作成する関数
    def make_dataset(root_path):
        #データセットのルートフォルダのPathオブジェクトを作成
        data_root = pathlib.Path(root_path)

        #データセットのラベル名(フォルダ名)を取得
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

        #データのリストを作成してシャッフルする
        all_image_paths = list(data_root.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        random.shuffle(all_image_paths)

        #ラベルにインデックスを割り当てる
        label_to_index = dict((name, index) for index,name in enumerate(label_names))

        #全ての画像データにラベル付け
        #all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
        #                    for path in all_image_paths]

        #dataaugmentationを考慮したラベル付け
        all_image_labels = []
        for path in all_image_paths:
            label = label_to_index[pathlib.Path(path).parent.name]
            all_image_labels.append(label)
            #for augment in range(7):
            #    all_image_labels.append(label)
        np_labels = np.array(all_image_labels)

        #画像のデータセットを作成
        imgs = []
        for filename in all_image_paths:
            img = cv2.imread(filename,cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0 #normalization
            imgs.append(img)

            #回転で水増し
            #imgs.append(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE))
            #imgs.append(cv2.rotate(img,cv2.ROTATE_180))
            #imgs.append(cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE))
            #反転して回転
            #rev_img = cv2.flip(img,1) #左右反転
            #imgs.append(rev_img)
            #imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_CLOCKWISE))
            #imgs.append(cv2.rotate(rev_img,cv2.ROTATE_180))
            #imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_COUNTERCLOCKWISE))


        np_imgs = np.array(imgs)

        return np_imgs,np_labels

    train_before_images,train_before_labels = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000')
    train_images,train_labels = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000_cleansing')

    test_images = train_images[:int(len(train_images)*0.1)]
    train_images = train_images[int(len(train_images)*0.1):]
    test_labels = train_labels[:int(len(train_labels)*0.1)]
    train_labels = train_labels[int(len(train_labels)*0.1):]

    train_before_images = train_before_images.reshape(-1,64,64,3)
    train_images = train_images.reshape(-1,64,64,3)

    score_mean,score_std = crossVal(train_images,train_labels)






