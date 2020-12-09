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

        #histogram equalization
        #clahe = cv2.createCLAHE(clipLimit=3,tileGridSize=(3,3))
        #img_v = clahe.apply(img_v)

        imgs.append(img_v)

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
    (test_images,test_labels) = make_dataset(r'E:\traning_data(murakami)\dataset_128px\test_data')
    test_imgs = np.copy(test_images)

    #normalization
    test_images = test_images / 255.0

    #reshape
    test_images = test_images.reshape(-1,128,128,1)
    #test_images = test_images.reshape(-1,64,64,1)

    #setting model
    model = keras.models.load_model(r'C:\Users\VIgpu01\workspace(murakami)\grid_search_result\result20201204.h5')

    model.summary()

    #predict
    predict_prob=model.predict(test_images)
    predict_classes=np.round(predict_prob).astype(int)

    #save error image
    i=0
    for label,predict in zip(test_labels,predict_classes):
        if predict != label:
            if label == 0:
                cv2.imwrite(r"C:\Users\VIgpu01\workspace(murakami)\WoodFeature\VSProjects\predict_test\predict_test\mistaking_high_for_low" + r"\error_%03.f"%(i) + ".tif",test_imgs[i])
            if label == 1:
                cv2.imwrite(r"C:\Users\VIgpu01\workspace(murakami)\WoodFeature\VSProjects\predict_test\predict_test\mistaking_low_for_high" + r"\error_%03.f"%(i) + ".tif",test_imgs[i])
        i+=1



    #Evaluate
    test_loss, test_acc = model.evaluate(test_images, test_labels, batch_size=128)
    
    print("test_loss:{}".format(test_loss))
    print("test_acc :{}".format(test_acc))
