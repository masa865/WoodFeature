#TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

#import scikit-learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#----------------データセット作成部-----------------------------------------------

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
        for augment in range(7):
            all_image_labels.append(label)
    np_labels = np.array(all_image_labels)

    #画像のデータセットを作成
    imgs = []
    for filename in all_image_paths:
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0 #normalization
        imgs.append(img)

        #回転で水増し
        imgs.append(cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE))
        imgs.append(cv2.rotate(img,cv2.ROTATE_180))
        imgs.append(cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE))
        #反転して回転
        rev_img = cv2.flip(img,1) #左右反転
        imgs.append(rev_img)
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_CLOCKWISE))
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_180))
        imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_COUNTERCLOCKWISE))


    np_imgs = np.array(imgs)

    return np_imgs,np_labels

#訓練データセット作成
train_images,train_labels = make_dataset(r'E:\traning_data(murakami)\yr_dataset_1000_cleansing')
#テストデータセット作成
#test_images,test_labels = make_dataset(r'E:\traning_data(murakami)\yr_dataset_test\test')

#-----------------------------------------------------------------------------------


#----------------学習モデル構築部-----------------------------------------------------
#モデルに入力するデータの整形
train_images = train_images.reshape(-1,64,64,3)
#test_images = test_images.reshape(-1,64,64,3)

#学習モデル
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

#モデルのコンパイル
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#モデルの学習
history = model.fit(train_images, train_labels, 
          batch_size=20,epochs=100,
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

#plot roc curve
y_pred_keras = model.predict(test_images).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

#テストデータを用いたモデルの評価
#test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=2)
#print('accuracy:')
#print(test_acc)
#-----------------------------------------------------------------------------------

