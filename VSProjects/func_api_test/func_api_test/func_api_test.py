# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

#scikit-learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#normalization
train_images = train_images / 255.0
test_images = test_images / 255.0

#reshape
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)

#learning model
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

#compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#learn model
#model.fit(train_images, train_labels, epochs=5)
history = model.fit(train_images, train_labels, 
          epochs=5,
          validation_data=(test_images,test_labels))

#plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss(binary cross enrropy)')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


#evaluate model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
