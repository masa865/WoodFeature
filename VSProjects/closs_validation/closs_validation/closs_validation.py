#cross validation
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import numpy as np
import pathlib
import random

import cv2

def cross_val(model,train_images,train_labels,ep=50,batchsize=32):
    kf = KFold(n_splits=5, shuffle=True)

    all_loss=[]
    all_val_loss=[]
    all_acc=[]
    all_val_acc=[]
    
    for train_index,val_index in kf.split(train_images,train_labels):
        train_data=train_images[train_index]
        train_label=train_labels[train_index]
        val_data=train_images[val_index]
        val_label=train_labels[val_index]

        history=model.fit(train_data,
                      train_label,
                      epochs=ep,
                      batch_size=batchsize,
                      validation_data=(val_data,val_label))

        loss=history.history['loss']
        val_loss=history.history['val_loss']
        acc=history.history['accuracy']
        val_acc=history.history['val_accuracy']

        all_loss.append(loss)
        all_val_loss.append(val_loss)
        all_acc.append(acc)
        all_val_acc.append(val_acc)

        ave_all_loss=[
            np.mean([x[i] for x in all_loss]) for i in range(ep)]
        ave_all_val_loss=[
            np.mean([x[i] for x in all_val_loss]) for i in range(ep)]
        ave_all_acc=[
            np.mean([x[i] for x in all_acc]) for i in range(ep)]
        ave_all_val_acc=[
            np.mean([x[i] for x in all_val_acc]) for i in range(ep)]

    return ave_all_loss,ave_all_acc,ave_all_val_loss,ave_all_val_acc


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

    model.summary()

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    before_loss,before_acc,val_before_loss,val_before_acc = cross_val(model,train_before_images,train_before_labels,ep=5,batchsize=32)
    loss,acc,val_loss,val_acc = cross_val(model,train_images,train_labels,ep=50,batchsize=32)

    #plot training & validation loss values
    plt.plot(before_loss,color='blue',  linestyle='solid')
    plt.plot(val_before_loss,color='blue',  linestyle='dashed')

    plt.plot(loss,color='red',  linestyle='solid')
    plt.plot(val_loss,color='red',  linestyle='dashed')

    plt.title('Model loss')
    plt.ylabel('Loss(binary cross entropy)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val','Cleansed_Train','Cleansed_Val'], loc='upper left')
    plt.show()

    #plot training & validation acces
    plt.plot(before_acc,color='blue',linestyle='solid')
    plt.plot(val_before_acc,color='blue',linestyle='dashed')

    plt.plot(acc,color='red',  linestyle='solid')
    plt.plot(val_acc,color='red',  linestyle='dashed')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val','Cleansed_Train','Cleansed_Val'], loc='upper left')
    plt.show()

    #plot roc curve
    y_pred_keras = model.predict(test_images).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Cleansed data(area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()






