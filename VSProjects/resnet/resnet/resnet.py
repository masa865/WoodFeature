#ResNetに関するモジュール

import tensorflow as tf
from tensorflow import keras

#Res Block
def rescell(data, filters, kernel_size, option=False):
    strides=(1,1)
    if option:
        strides=(2,2)
    x=keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(data)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Activation('relu')(x)
    data=keras.layers.Conv2D(filters=int(x.shape[3]), kernel_size=(1,1), strides=strides, padding="same")(data)
    x=keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=(1,1),padding="same")(x)
    x=keras.layers.BatchNormalization()(x)

    x=keras.layers.Add()([x,data])
    x=keras.layers.Activation('relu')(x)

    return x

#ResNet34(for 10 classes classfication)
def resnet(img_rows,img_cols,img_channels,train_img):
	input=keras.layers.Input(shape=(img_rows,img_cols,img_channels))
	
	x=keras.layers.Conv2D(32,(7,7), padding="same", input_shape=train_img.shape[1:],activation="relu")(input)
	x=keras.layers.MaxPooling2D(pool_size=(2,2))(x)

	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))
	x=rescell(x,64,(3,3))

	x=rescell(x,128,(3,3),True)

	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))
	x=rescell(x,128,(3,3))

	x=rescell(x,256,(3,3),True)

	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))
	x=rescell(x,256,(3,3))

	x=rescell(x,512,(3,3),True)

	x=rescell(x,512,(3,3))
	x=rescell(x,512,(3,3))

	x=keras.layers.AveragePooling2D(pool_size=(int(x.shape[1]),int(x.shape[2])),strides=(2,2))(x)

	x=keras.layers.Flatten()(x)
	x=keras.layers.Dense(units=10,kernel_initializer="he_normal",activation="softmax")(x)
	model=keras.Model(inputs=input,outputs=[x])

	return model


#test script
if __name__ == "__main__":
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
	#ResNet34
	res_model = resnet(28,28,1,train_images)
	#Mymodel
	my_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])



	#compile model
	res_model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	my_model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])

	#learn model
	#model.fit(train_images, train_labels, epochs=5)
	res_history = res_model.fit(train_images, train_labels, 
			  batch_size=128,epochs=50,
			  validation_data=(test_images,test_labels))
	my_history = my_model.fit(train_images, train_labels, 
			  batch_size=128,epochs=50,
			  validation_data=(test_images,test_labels))

	#plot training & validation loss values
	plt.plot(res_history.history['loss'],color='red')
	plt.plot(res_history.history['val_loss'],'--',color='red')
	plt.plot(my_history.history['loss'],color='blue')
	plt.plot(my_history.history['val_loss'],'--',color='blue')
	#plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['ResNet Train', 'ResNet Val','My Train', 'My Val'], loc='upper left')
	plt.show()


	#evaluate model
	#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)