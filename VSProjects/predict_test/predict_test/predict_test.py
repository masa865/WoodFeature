import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import random
import cv2

#function that creates a dataset labeled with the folder name under root_path
def make_dataset(root_path,data_augmentation=False):
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

        #if(data_augmentation):
        #    for augment in range(7):
        #        all_image_labels.append(label)

    np_labels = np.array(all_image_labels) #convert to numpy array

    #make image dataset 
    imgs = []
    for filename in all_image_paths:
        img = cv2.imread(filename,cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_h,img_s,img_v = cv2.split(img_hsv)
        imgs.append(img_v)

        #histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(3,3))
        img_v = clahe.apply(img_v)

        #if(data_augmentation):
        #    #augmentation(rotate)
        #    imgs.append(cv2.rotate(img_v,cv2.ROTATE_90_CLOCKWISE))
        #    imgs.append(cv2.rotate(img_v,cv2.ROTATE_180))
        #    imgs.append(cv2.rotate(img_v,cv2.ROTATE_90_COUNTERCLOCKWISE))
            #augmentation(reverse)
        #    rev_img = cv2.flip(img_v,1) #左右反転
        #    imgs.append(rev_img)
        #    imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_CLOCKWISE))
        #    imgs.append(cv2.rotate(rev_img,cv2.ROTATE_180))
        #    imgs.append(cv2.rotate(rev_img,cv2.ROTATE_90_COUNTERCLOCKWISE))

    np_imgs = np.array(imgs) #convert to numpy array

    return (np_imgs,np_labels)



#test code for this module
if __name__ == '__main__':

    from sklearn.model_selection import KFold
    from sklearn.model_selection import train_test_split
    
    #import data
    (test_images,test_labels) = make_dataset(r'E:\traning_data(murakami)\test_data_50012')
    test_imgs = np.copy(test_images)

    #normalization
    test_images = test_images / 255.0

    #reshape
    test_images = test_images.reshape(-1,64,64,1)

    #setting model
    #model = load_model('./フォルダ名/' + model_file_name+'.h5')

    model.summary()

    #predict
    predicts = model.predict_classes(test_images)

    #save error image
    i=0
    for img,label,predict in test_imgs,test_labels,predicts:
        if predict != label:
            if label == 0:
                cv2.imwrite(r"C:\Users\sirim\Pictures\outdoor1_tif_DSC_0573_trim\denoising\splited" + r"\error_%03.f"%(i) + ".tif",img)
            if label == 1:
                cv2.imwrite(r"C:\Users\sirim\Pictures\outdoor1_tif_DSC_0573_trim\denoising\splited" + r"\error_%03.f"%(i) + ".tif",img)
            i+=1



    #Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=128)
    
    print("test_loss:{}".format(test_loss))
    print("test_acc :{}".format(test_acc))
