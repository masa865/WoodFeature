#TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

#import scikit-learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#helper library
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#my module
from dataset import make_dataset


#----------------データセット作成部-----------------------------------------------
train_images,train_labels,test_images,test_labels = make_dataset(r'hoge')
#-----------------------------------------------------------------------------------


#----------------学習モデル構築部-----------------------------------------------------
#モデルに入力するデータの整形
train_images = train_images.reshape(-1,64,64,3)
test_images = test_images.reshape(-1,64,64,3)

#学習モデル
#learning model
inputs = keras.layers.Input(shape=(64,64,3))
x = keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
out = keras.layers.Dense(1, activation='softmax')(x)

model = keras.Model(inputs=inputs,outputs=out)

model.summary()


#モデルのコンパイル
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#モデルの学習
history = model.fit(train_images, train_labels, 
          epochs=5,
          validation_data=(test_images,test_labels))

#plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss(binary cross enrropy)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
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

