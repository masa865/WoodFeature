#for Features extraction of wood
#import tensorflow as tf
#from tensorflow import keras

import cv2

import numpy as np
import math

#split image
def splitImg(img,split_size=128):

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
    extractable_flag = 1

    k = 0
    for i in range(v_split):
        for j in range(h_split):
            if(predict_results[j+k]):
                flag_img[i*split_size:i*split_size+split_size,
                         j*split_size:j*split_size+split_size] = extractable_flag
        k+=h_split

    return flag_img

#calculate coordinate of circle
def getCircleXY(radius,center_x,center_y):

    X_up = np.arange(center_x - radius, center_x + radius + 1)
    X = np.append(X_up,X_up[1:len(X_up)-1])
    Y_up = np.sqrt(radius**2 - (X_up-center_x)**2) + center_y
    Y_down = -Y_up + 2*center_y
    Y = np.append(Y_up,Y_down[1:len(Y_down)-1])

    return (X,Y) #X,Y are numpy array

#obtain edges of the annual rings
def obtainEdges(img,minVal=10,maxVal=50,filter_size=3):
    #minVal=100,maxVal=200,filter_size=3
    img_edge = cv2.Canny(img,minVal,maxVal,filter_size)

    return img_edge

def getNR(img,center_x,center_y,outerX,outerY):
    #---------------------------------------------
    #NR  :number of annual rings
    #AR  :average of every ring(px)
    #AC15:average width of 15th from the center(px)
    #AO15: average width of 15th from the outside(px)
    #img : assumed edge image
    #ring_nums[line_index] : ring num of "line_index" th line
    #---------------------------------------------

    ring_nums = np.zeros_like(outerX)
    line_index = 0

    img_copy = np.copy(img)

    for outerx,outery in list(zip(outerX,outerY)):

        if(center_x-outerx != 0): #intersept is not infinity
            if (outerx > center_x):
                X = np.arange(center_x, outerx+0.1,0.1)
            if(outerx < center_x):
                X = np.arange(outerx,center_x+0.1,0.1)

            intersept = (center_y - outery) / (center_x - outerx)
            Y = intersept*(X-center_x)+center_y
            same_line_flag = False

            if(abs(Y[0]-Y[1]) < 1.5): #distance between the points is not too large
                for x,y in list(zip(X,Y)):
                    img_copy[math.ceil(x),math.ceil(y)]=120
                    if(img[math.ceil(x),math.ceil(y)] != 0):
                        if(same_line_flag == False):
                            ring_nums[line_index] += 1
                            same_line_flag = True
                    else:
                        same_line_flag = False
        line_index += 1

    cv2.imshow("img_copy",img_copy)
    cv2.waitKey(0)

    NR = np.floor(ring_nums[np.nonzero(ring_nums)].mean())
    print("-result--------")
    print("ring_nums:")
    print(ring_nums)
    print("NR(floor):{}".format(NR))
    print("NR(float):{}".format(ring_nums[np.nonzero(ring_nums)].mean()))
    return NR

#main function of this module
def extractFeature(img,center_x,center_y,radius,model):
    #img:v channel of hsv

    #----------------------------------------------------------------------------
    #---1.extract Low-noise line--------------------------------------------------
    #----------------------------------------------------------------------------
    split_size = 128 #same size as model input
    splited_imgs,v_split,h_split = splitImg(img,split_size)
    
    #prediction using model
    predictions = model.predict(splited_imgs)
    predict_results = predictions.argmax(axis=1) #result list of classification

    #create flag image
    flag_img = createFlag(img,v_split,h_split,predict_results,split_size)

    #get coordinate of outer wood
    (outerX,outerY) = getCircleXY(radius,center_x,center_y)

    #extract good line
    line_values = np.zeros_like(outerX)
    line_index = 0
    for outerx,outery in outerX,outerY:

        if(center_x-outerx != 0): #intersept is not infinity
            if (outerx > center_x):
                X = np.arange(center_x, outerx+0.1,0.1)
            if(outerx < center_x):
                X = np.arange(outerx,center_x+0.1,0.1)

            intersept = (center_y - outery) / (center_x - outerx)
            #print(intersept)
            Y = intersept*(X-center_x)+center_y

            if(abs(Y[0]-Y[1]) < 1.5): #distance between the points is not too large
                for x,y in list(zip(X,Y)):
                    if(img[math.ceil(x),math.ceil(y)] != 0):
                        line_values[line_index] += 1
        line_index += 1

    #get good line indexes(prototype criteria)
    median = np.median(line_values)
    good_line_indexes = np.where(line_values >= median)[0]

    good_outerX = []
    good_outerY = []
    for index in good_line_indexes:
        good_outerX.append(outerX[index])
        good_outerY.append(outerY[index])

    #----------------------------------------------------------------------------
    #---2.obtain edge image------------------------------------------------------
    #----------------------------------------------------------------------------
    img_edge = obtainEdges(img)

    #----------------------------------------------------------------------------
    #---3.calculate NR-----------------------------------------------------------
    #----------------------------------------------------------------------------
    NR = getNR(img_edge,center_x,center_y,good_outerX,good_outerY)

    #calculate AR

    #calculate AC15

    #calculate AO15

    return NR,AR,AC15,AO15

def extractByTraditional(img,center_x,center_y,radius):
    #img:v channel of hsv
    NR=0
    AR=0
    AC15=0
    AO15=0

    #get coordinate of outer wood
    (outerX,outerY) = getCircleXY(radius,center_x,center_y)

    #1.obtain edge image
    img_edge = obtainEdges(img)

    cv2.imshow('img_edge',img_edge)
    cv2.waitKey(0)

    #2.calculate NR
    NR = getNR(img_edge,center_x,center_y,outerX[::10],outerY[::10])

    return NR,AR,AC15,AO15

#test code for this module
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    load_img = cv2.imread(r"C:\Users\sirim\Pictures\indoor_denoised\50012.tif",0)

    img_edge = obtainEdges(load_img)
    cv2.imshow("img_edge",img_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #load_img,172,185,308-160
    #NR,AR,AC15,AO15=extractByTraditional(load_img,180,170,308-160)


    #center_x = 172
    #center_y = 185
    #(outerX,outerY) = getCircleXY(308-160,172,185)
    #plt.plot(outerX, outerY,marker='.',linestyle='None')

    #for outerx,outery in list(zip(outerX[::10],outerY[::10])):
    #    if(center_x-outerx != 0):
    #        if (outerx > center_x):
    #            X = np.arange(center_x, outerx+0.1,0.1)
    #        if(outerx < center_x):
    #            X = np.arange(outerx,center_x+0.1,0.1)

    #        intersept = (center_y - outery) / (center_x - outerx)
    #        print(intersept)
    #        Y = intersept*(X-center_x)+center_y

    #        if(outery>center_y):np.clip(Y, None, outery)
    #        else: np.clip(Y,outery, None)

    #        plt.plot(X, Y,marker='.',linestyle='None')

    #plt.axis("equal")
    #plt.grid(color="0.8")
    #plt.show() # 画面に表示

    #cv2.imshow("result",img_edge)
    #cv2.waitKey(0)
