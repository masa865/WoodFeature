import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
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
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        
        img_v = np.array((img_v - np.mean(img_v)) / np.std(img_v) * 20 + 148,dtype=np.uint8) #normalization
        img_v = np.clip(img_v,0,255)

        #img_v = cv2.equalizeHist(img_v)

        imgs.append(img_v)

    np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)


#test code for this module
if __name__ == '__main__':

    folder_path = r'E:\traning_data(murakami)\yr_dataset_1000_cleansing\low'

    data_root = pathlib.Path(folder_path)

    all_image_paths = list(data_root.glob('*'))
    all_image_paths = [str(path) for path in all_image_paths]

    imgs = []
    for filename in all_image_paths:
        print(filename)
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)

    imgs.append(img_v)

    np_imgs = np.array(imgs) #convert to numpy array

    print("np_imgs mean:{}".format(np.mean(np_imgs)))
    print("np_imgs std :{}".format(np.std(np_imgs)))

    #150
    #16