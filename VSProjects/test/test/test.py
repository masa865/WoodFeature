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

#訓練データセット作成
#データセットのルートフォルダのPathオブジェクトを作成
data_root = pathlib.Path(r'E:\traning_data(murakami)\yr_dataset_test\train')
#print(data_root)

#データセットのラベル名(フォルダ名)を取得
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
#print(label_names)

#データのリストを作成してシャッフルする
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
#print(all_image_paths)

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

BATCH_SIZE = 64
image_count = 1000
# シャッフルバッファのサイズをデータセットとおなじに設定することで、データが完全にシャッフルされる
# ようにできます。
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch`を使うことで、モデルの訓練中にバックグラウンドでデータセットがバッチを取得できます。
ds = ds.prefetch(buffer_size=AUTOTUNE)

#テストデータセット作成

#-----------------------------------------------------------------------------------


#----------------学習モデル構築部-----------------------------------------------------

#-----------------------------------------------------------------------------------