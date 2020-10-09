# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.experimental.AUTOTUNE

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#----------------データセット作成部-----------------------------------------------

#root_path内のデータのデータセット作成
def make_dataset(root_path):
    #データセットのルートフォルダのPathオブジェクトを作成
    data_root = pathlib.Path(root_path)

    #データセットのラベル名(フォルダ名)を取得
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    #データのリストを作成してシャッフルする
    all_image_paths = list(data_root.glob('*/*'))
    print(all_image_paths)
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    #実験のために使う画像数をall_numからneed_numに減らす
    need_num = 1000
    all_num = len(all_image_paths)
    all_image_paths[need_num:all_num] = []


    #ラベルにインデックスを割り当てる
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    #全ての画像データにラベル付け
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in all_image_paths]

    #画像のデータセットを作成
    imgs = []
    for filename in all_image_paths:
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        imgs.append(img)
    
    image_ds = tf.data.Dataset.from_tensor_slices(imgs)
    #ラベルのデータセットを作成
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
    #(画像,ラベル)のペアのデータセットを作成
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    return image_label_ds

#訓練データセット作成
image_label_train_ds = make_dataset(r'E:\traning_data(murakami)\yr_dataset_test\train')
#テストデータセット作成
test_ds = make_dataset(r'E:\traning_data(murakami)\yr_dataset_test\test')

#訓練データで学習しやすくする処理
BATCH_SIZE = 64
#image_count = 1000
# シャッフルバッファのサイズをデータセットとおなじに設定することで、データが完全にシャッフルされる
# ようにできます。
train_ds = image_label_train_ds.shuffle(buffer_size=image_count)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
# `prefetch`を使うことで、モデルの訓練中にバックグラウンドでデータセットがバッチを取得できます。
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
#-----------------------------------------------------------------------------------


#----------------学習モデル構築部-----------------------------------------------------
#学習モデル
print('訓練データセットのshape:')
print(train_ds)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model.summary()

#モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#モデルの学習
model.fit(train_ds,epochs=5,steps_per_epoch=3)

#テストデータを用いたモデルの評価
test_loss, test_acc = model.evaluate(test_ds,verbose=2)
print('accuracy:')
print(test_acc)
#-----------------------------------------------------------------------------------