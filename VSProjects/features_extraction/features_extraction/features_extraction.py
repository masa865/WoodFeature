#for Features extraction of wood
import tensorflow as tf
from tensorflow import keras

import cv2

import numpy as np

#calculate parameter of line equation from 2 point
def lineEquation(x0,y0,xn,yn):

    if xn - x0 == 0: #if intersept = infinity 
        return (float('inf'), None)
    
    intersept = (y0 - yn) / (x0 - xn)
    slope = y0 - intersept * x0

    return intersept,slope

#calculate coordinate of circle
def getCircleXY(radius,center=(0,0),divn=10):

    theta = np.linspace(0,2*np.pi,divn) #radian

    X = center[0] + radius * np.cos(theta) #X is ndarray of x coordinate
    Y = center[1] + radius * np.sin(theta) #Y is ndarray of y coordinate

    return X,Y

#calculate num of annual rings
def getRingNum(outerX,outerY,center=(0,0)):

    x0,y0 = center

    outerX_n = outerX.size
    for i in outerX_n:
        intersept = (y0 - outerX[i]) / (x0 - outerY[i])
        num = 0




#obtain edges of the annual rings
def obtainEdges(img_gray,minVal=100,maxVal=200,filter_size=3):

    img_edge = cv2.Canny(img_gray,minVal,maxVal,filter_size)

    return img_edge

def splitImg(img,split_size=64):

    v_size = img.shape[0] // size * size
    h_size = img.shape[1] // size * size
    img = img[:v_size, :h_size]

    v_split = img.shape[0] // size #split num of vertical direction
    h_split = img.shape[1] // size #split num of horizontal direction

    i=0
    j=0
    for h_img in np.vsplit(img, v_split):
        for v_img in np.hsplit(h_img, h_split):
                #cv2.imwrite(r"C:\Users\user\Pictures\opencv_test_pic\test" + r"\img_%03.f"%(i) + "_%03.f"%(j) + ".jpg",v_img)
                cv2.imwrite(r"C:\Users\sirim\Pictures\new\output\test_image" + r"\img_%03.f_%03.f"%(i,j) + ".tif",v_img)
                j+=1 #update row index
        j=0 #reset row index
        i+=1 #update col index

    return splited_imgs,splited_index

#main function of this module
def extractFeature(img,x_center,y_center,radius,model):
    #img:v channel of hsv

    #select Low-noise line
    #split img
    split_size = 64

    
    #predict
    predictions = model.predict(splited_imgs)
    predict_results = predictions.argmax(axis=1) #result list of classification

    #create flag img
    flag_img = numpy.zeros_like(img)
    class_dict = {"high":0,"low":1}
    

    #obtain edge image
    img_edge = obtainEdges(img)

    #calculate NR

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

    #load_img = cv2.imread(r"C:\Users\sirim\Pictures\new\DSC_0547_1_8.tif",cv2.IMREAD_GRAYSCALE)

    #img_blur = cv2.medianBlur(load_img,3)

    #edge = obtainEdges(img_blur)

    #cv2.imshow('load_img',load_img)
    #cv2.waitKey(0)
    #cv2.imshow('img_blur',img_blur)
    #cv2.waitKey(0)
    #cv2.imshow('edge',edge)
    #cv2.waitKey(0)

    x,y = getCircleXY(2,(100,100))
    print(x)
    print(y)



