#機械学習に使うデータセットを作成するためのモジュール

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

#root_path内のデータのデータセット作成
def make_dataset(root_path,train_data_ratio=0.8):
    #データセットのルートフォルダのPathオブジェクトを作成
    data_root = pathlib.Path(root_path)

    #データセットのラベル名(フォルダ名)を取得
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

    #データのリストを作成してシャッフルする
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    #学習に使うため画像の枚数をall_numからneed_numに調節
    need_num = 100
    all_num = len(all_image_paths)
    all_image_paths[need_num:all_num] = []

    train_paths = all_image_paths[:int(len(all_image_paths)*train_data_ratio)]
    test_paths = all_image_paths[int(len(all_image_paths)*train_data_ratio):]

    #ラベルにインデックスを割り当てる
    label_to_index = dict((name, index) for index,name in enumerate(label_names))

    #ラベル付け
    tr_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in train_paths]
    train_labels = np.array(tr_labels)
    ts_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in test_paths]
    test_labels = np.array(ts_labels)

    #画像のデータセットの作成
    train_imgs = []
    for filename in train_paths:
        train_img = cv2.imread(filename,cv2.IMREAD_COLOR)
        train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
        train_img = train_img / 255.0
        train_imgs.append(train_img)
    train_images = np.array(train_imgs)
    test_imgs = []
    for filename in test_paths:
        test_img = cv2.imread(filename,cv2.IMREAD_COLOR)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = test_img / 255.0
        test_imgs.append(test_img)
    test_images = np.array(test_imgs)

    return train_images,train_labels,test_images,test_labels


#----------------test code-------------------------------------
if __name__ == "__main__":
    #データセット作成
    train_images,train_labels,test_images,test_labels = make_dataset(r'C:\traning_data(murakami)\yr_dataset')

    print(train_images)
    print(train_labels)
    print(test_images)
    print(test_labels)


