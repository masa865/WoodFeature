#for Features extraction of wood
import tensorflow as tf
from tensorflow import keras

import cv2

import numpy as np
import math

#split image
def splitImg(img,split_size=64):

    v_size = img.shape[0] // split_size * split_size
    h_size = img.shape[1] // split_size * split_size
    img = img[:v_size, :h_size]

    v_split = img.shape[0] // split_size #split num of vertical direction
    h_split = img.shape[1] // split_size #split num of horizontal direction

    splited_imgs = []
    i=0
    j=0
    for h_img in np.vsplit(img, v_split):
        for v_img in np.hsplit(h_img, h_split):
                splited_imgs.append(v_img)
                j+=1 #update horizontal direction index
        j=0 #reset horizontal direction index
        i+=1 #update vertical direction index
    

    return np.array(splited_imgs),v_split,h_split

#create flag image
def createFlag(img,v_split,h_split,predict_results,split_size=64):

    flag_img = np.zeros_like(img)
    k = 0
    for i in range(v_split):
        for j in range(h_split):
            if(predict_results[j+k]):
                flag_img[i*split_size:i*split_size+split_size,
                         j*split_size:j*split_size+split_size] = 1
        k+=h_split

    return flag_img

#calculate coordinate of circle
def getCircleXY(radius,center=(0,0),divn=10):

    theta = np.linspace(0,2*np.pi,divn) #radian

    X = center[0] + radius * np.cos(theta) #X is ndarray of x coordinate
    Y = center[1] + radius * np.sin(theta) #Y is ndarray of y coordinate

    return (X,Y)

#obtain edges of the annual rings
def obtainEdges(img,minVal=100,maxVal=200,filter_size=3):

    img_edge = cv2.Canny(img,minVal,maxVal,filter_size)

    return img_edge

def getNR(img,center_x,center_y,outerX,outerY):
    #img:edge image

    ring_nums = np.zeros_like(outerX)
    line_index = 0
    for outerx,outery in outerX,outerY:
        intersept = (center_y - outery) / (center_x - outerx)
        x = math.ceil(outerx)
        while x != center_x:
            print("loop checker")
            y = math.ceil(intersept*(x-center_x)+senter_y)
            if(img[x,y]==1):
                ring_nums[line_index]+=1
        line_index+=1

    NR = np.floor(np.mean(ring_nums))

    return NR

#main function of this module
def extractFeature(img,center_x,center_y,radius,model):
    #img:v channel of hsv

    #1.select Low-noise line
    #split img
    split_size = 64
    splited_imgs,v_split,h_split = splitImg(img,split_size)
    
    #predict
    predictions = model.predict(splited_imgs)
    predict_results = predictions.argmax(axis=1) #result list of classification

    #create flag img
    flag_img = createFlag(img,v_split,h_split,predict_results,split_size)

    #get coordinate of outer wood
    (outerX,outerY) = getCircleXY(radius,(center_x),(center_y),divn=20)

    #extract good line
    line_values = np.zeros_like(outerX)
    line_index = 0
    for outerx,outery in outerX,outerY:
        intersept = (center_y - outery) / (center_x - outerx)
        x = math.ceil(outerx)
        while x != center_x:
            print("loop checker")
            y = math.ceil(intersept*(x-center_x)+senter_y)
            if(flag_img[x,y]==1):
                line_values[line_index]+=1
        line_index+=1

    #get good line indexes(prototype criteria)
    median = np.median(line_values)
    good_line_indexes = np.where(line_values >= median)[0]

    #2.obtain edge image
    img_edge = obtainEdges(img)

    #3.calculate NR
    NR = getNR(img_edge,center_x,center_y,good_outerX,good_outerY)

    #calculate AR

    #calculate AC15

    #calculate AO15

    #---------------------------------------------
    #NR  :number of annual rings
    #AR  :average of every ring(cm)
    #AC15:average width of 15th from the center(cm)
    #AO15: average width of 15th from the outside(cm)
    #---------------------------------------------
    return NR,AR,AC15,AO15

#test code for this module
if __name__ == '__main__':

    load_img = cv2.imread(r"C:\Users\sirim\Pictures\new\deru.jpg",cv2.IMREAD_GRAYSCALE)

    cv2.imshow('load_img',load_img)
    cv2.waitKey(0)

    #hist_img = cv2.equalizeHist(load_img)
    hist_img = np.array((load_img - np.mean(load_img)) / np.std(load_img) * 16 + 128,dtype=np.uint8) #normalization
    hist_img = np.clip(hist_img, 0, 255)
    cv2.imshow('hist_img',hist_img)
    cv2.waitKey(0)

    #img_blur = cv2.medianBlur(load_img,3)

    #edge = obtainEdges(img_blur)

    #cv2.imshow('load_img',load_img)
    #cv2.waitKey(0)
    #cv2.imshow('img_blur',img_blur)
    #cv2.waitKey(0)
    #cv2.imshow('edge',edge)
    #cv2.waitKey(0)

    #x,y = getCircleXY(2,(100,100))
    #print(x)
    #print(y)

    model = tf.keras.models.load_model('test_model_73_rotrev.h5')
    model.summary()

    splited_imgs,v_split,h_split = splitImg(load_img)
    splited_imgs = splited_imgs /255.0
    splited_imgs = splited_imgs.reshape(-1,64,64,1)

    predictions = model.predict(splited_imgs)
    predict_results = predictions.argmax(axis=1) #result list of classification
    #print(predict_results)

    flag_img = createFlag(load_img,v_split,h_split,predict_results,split_size=64)

    print(flag_img.shape)

    line_values = np.array([10,11,12,13,14,15,16,17,18,19,20])
    median = np.median(line_values)
    good_line_indexes = np.where(line_values >= median)
    print("median:{}".format(median))
    print(good_line_indexes)
    print(good_line_indexes[0])

    #print(splited_imgs.shape)
    #cv2.imshow('splited_imgs',splited_imgs[0])
    #cv2.waitKey(0)



