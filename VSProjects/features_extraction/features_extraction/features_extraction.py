#for Features extraction of wood
import tensorflow as tf
from tensorflow import keras

import cv2

import matplotlib.pyplot as plt
import numpy as np
import math
import time
import sys 
import csv
import os
import pathlib

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
def createFlag(img,v_split,h_split,predict_results,split_size=128):

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

def swap(n1,n2):

    return [n2,n1]

#calculate line with the Bresenham algorithm
def getLineXY(start,end):
    #start=[x1,y1],end=[x2,y2]

    lineX = []
    lineY = []
    start_tmp = start[:]
    end_tmp = end[:]

    steep = abs(end[1]-start[1]) > abs(end[0]-start[0])
    if steep:
        start = swap(start[0],start[1])
        end = swap(end[0],end[1])

    if start[0] > end[0]:
        _x0 = int(start[0])
        _x1 = int(end[0])
        start[0] = _x1
        end[0] = _x0

        _y0 = int(start[1])
        _y1 = int(end[1])
        start[1] = _y1
        end[1] = _y0

    dx = end[0] - start[0]
    dy = abs(end[1] - start[1])
    error = 0
    derr = dy/float(dx)

    ystep = 0
    y = start[1]

    if start[1] < end[1]: ystep = 1
    else: ystep = -1

    for x in range(int(start[0]),int(end[0])+1):
        if steep:
            lineX.append(y)
            lineY.append(x)
        else:
            lineX.append(x)
            lineY.append(y)

        error += derr

        if error >= 0.5:
            y += ystep
            error -= 1.0

    #align the starting point
    dist1 = np.sqrt((lineX[0]-start_tmp[0])**2 + (lineY[0]-start_tmp[1])**2)
    dist2 = np.sqrt((lineX[-1]-start_tmp[0])**2 + (lineY[-1]-start_tmp[1])**2)
    if dist1 > dist2 :
        lineX.reverse()
        lineY.reverse()

    return (np.array(lineX),np.array(lineY))

#calculate coordinate of circle
def getCircleXY(radius,center_x,center_y,n_points=720):

    theta = np.arange(0, 2 * np.pi, (2 * np.pi)/n_points)

    X =center_x + radius * np.sin(theta)
    Y =center_y + radius * np.cos(theta)

    return (X,Y) #X,Y are numpy array

#obtain edges of the annual rings
def obtainEdges(img,minVal=130,maxVal=130,filter_size=3):

    print("debug> Canny param:minVal={},maxVal={}".format(minVal,maxVal))
    img_edge = cv2.Canny(img,minVal,maxVal,filter_size,)

    return img_edge

def calcFeatures(img,center_x,center_y,outerX,outerY):
    #---------------------------------------------
    #NR  :number of annual rings
    #AR  :average of every ring(px)
    #AC15:average width of 15th from the center(px)
    #AO15: average width of 15th from the outside(px)
    #img : assumed edge image
    #ring_nums[line_index] : ring num of "line_index" th line
    #---------------------------------------------
    ring_nums = np.zeros_like(outerX)
    ring_widths = []
    ring_pos = [] #ring positions
    ac15_array = []
    ao15_array = []
    ar_array = []
    line_index = 0
    dist_th = 1.5 #distance threshold
    im_height,im_width = img.shape

    img_copy = np.copy(img)
    img_c = cv2.cvtColor(img_copy,cv2.COLOR_GRAY2BGR)

    for outerx,outery in list(zip(outerX,outerY)):
        
        (X,Y) = getLineXY([center_x,center_y],[outerx,outery])
        X[X >= im_width] = im_width - 1
        Y[Y >= im_height] = im_height - 1
        same_line_flag = False

        cv2.line(img_c,(center_x,center_y),(int(outerx),int(outery)),(0,0,255),thickness=3,lineType=cv2.LINE_AA)

        for x,y in list(zip(X,Y)):
            
            #img_c[y,x,2]=255

            if(img[y,x] != 0):
                if(same_line_flag == False):
                    ring_nums[line_index] += 1
                    ring_pos.append((x,y))

                    same_line_flag = True
            else:
                same_line_flag = False

        if(len(ring_pos) >= 2):
            sum = 0
            for i in range(len(ring_pos)-1):
                sum += math.sqrt(((ring_pos[i+1])[0] - (ring_pos[i])[0]) ** 2 + ((ring_pos[i+1])[1] - (ring_pos[i])[1]) ** 2)
            ar_array.append(sum/(len(ring_pos)-1))

        if(len(ring_pos) >= 15):
            sum = 0
            for i in range(14):
                sum += math.sqrt(((ring_pos[i+1])[0] - (ring_pos[i])[0]) ** 2 + ((ring_pos[i+1])[1] - (ring_pos[i])[1]) ** 2)
            ac15_array.append(sum/14)

            ring_pos_rev = ring_pos[::-1]
            sum = 0
            for i in range(14):
                sum += math.sqrt(((ring_pos_rev[i+1])[0] - (ring_pos_rev[i])[0]) ** 2 + ((ring_pos_rev[i+1])[1] - (ring_pos_rev[i])[1]) ** 2)
            ao15_array.append(sum/14)
        
        ring_pos = [] #reset ring_pos
        line_index += 1 #next line

    #cv2.namedWindow('img_c', cv2.WINDOW_KEEPRATIO)
    #cv2.imwrite(r'C:\Users\VIgpu01\Pictures\fe_result\line_trad.tif',img_c)
    #cv2.imwrite(r'C:\Users\sirim\Pictures\feat_result\line_trad.tif',img_c)
    #cv2.imshow("img_c",img_c)
    #cv2.waitKey(0)

    NR,AR,AC15,AO15 = 0,0,0,0
    if(len(ring_nums) > 0):NR = np.floor(ring_nums[np.nonzero(ring_nums)].mean())
    if(len(ar_array) > 0):AR = np.array(ar_array).mean()
    if(len(ac15_array) > 0):AC15 = np.array(ac15_array).mean()
    if(len(ao15_array) > 0):AO15 = np.array(ao15_array).mean()

    print("-result--------")
    print("ring_nums:")
    print(ring_nums[np.nonzero(ring_nums)])
    print("NR(floor):{}".format(NR))
    nr_std = np.std(ring_nums[np.nonzero(ring_nums)])
    print("NR(float):{:.2f}(±{:.2f})".format(ring_nums[np.nonzero(ring_nums)].mean(),nr_std))
    print("AR:{:.2f}px".format(AR))
    print("AC15:{:.2f}px".format(AC15))
    print("AO15:{:.2f}px".format(AO15))

    return NR,AR,AC15,AO15

#main function of this module
def extractFeature(img,center_x,center_y,radius,model,thresh=None):
    #img:v channel of hsv
    print("debug> mode:new method")
    img_tmp = np.copy(img)

    #----------------------------------------------------------------------------
    #---1.extract Low-noise line--------------------------------------------------
    #----------------------------------------------------------------------------
    split_size = 128 #same size as model input
    splited_imgs,v_split,h_split = splitImg(img,split_size)

    #prediction using model
    if(thresh != None):
        if(thresh == 'otsu'):
            for i in range(len(splited_imgs)):
                _,splited_imgs[i] = cv2.threshold(splited_imgs[i],0,255,cv2.THRESH_OTSU)
        if(thresh == 'adaptive'):
            for i in range(len(splited_imgs)):
                splited_imgs[i] = cv2.adaptiveThreshold(splited_imgs[i],255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,2)

    splited_imgs = splited_imgs / 255.0
    splited_imgs = splited_imgs.reshape(-1,128,128,1)
    predictions = model.predict(splited_imgs)

    flat=predictions.flatten()
    low_noise = 1
    high_noise = 0
    predict_results = np.where(flat>=0.5,low_noise,high_noise)

    print("debug> predict_results:{}".format(predict_results))

    #create flag image
    flag_img = createFlag(img,v_split,h_split,predict_results,split_size)

    #get coordinate of outer wood
    (outerX,outerY) = getCircleXY(radius,center_x,center_y)

    #extract good line
    line_values = np.zeros_like(outerX)
    line_index = 0
    dist_th = 1.5
    im_height,im_width = flag_img.shape
    for outerx,outery in list(zip(outerX,outerY)):

        (X,Y) = getLineXY([center_x,center_y],[outerx,outery])
        X[X >= im_width] = im_width - 1
        Y[Y >= im_height] = im_height - 1

        for x,y in list(zip(X,Y)):
            if(flag_img[math.ceil(y),math.ceil(x)] != 0):
                line_values[line_index] += 1

        line_index += 1

    #get good line indexes
    good_line_indexes = np.where(line_values > (radius//20))[0]

    good_outerX = []
    good_outerY = []
    for index in good_line_indexes:
        good_outerX.append(outerX[index])
        good_outerY.append(outerY[index])

    #----------------------------------------------------------------------------
    #---2.obtain edge image------------------------------------------------------
    #----------------------------------------------------------------------------
    img_edge = obtainEdges(img_tmp)

    #cv2.namedWindow('img_edge', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('img_edge',img_edge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #----------------------------------------------------------------------------
    #---3.calculate features-----------------------------------------------------------
    #----------------------------------------------------------------------------
    if(np.sum(good_line_indexes) != 0):
        NR,AR,AC15,AO15 = calcFeatures(img_edge,center_x,center_y,good_outerX,good_outerY)
    else:
        NR,AR,AC15,AO15 = calcFeatures(img_edge,center_x,center_y,outerX[::10],outerY[::10])

    return NR,AR,AC15,AO15

def extractByTraditional(img,center_x,center_y,radius):
    print("debug> mode:traditional method")

    #get coordinate of outer wood
    (outerX,outerY) = getCircleXY(radius,center_x,center_y)

    #1.obtain edge image
    img_edge = obtainEdges(img)

    #cv2.namedWindow('img_edge', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('img_edge',img_edge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #2.calculate feature
    NR,AR,AC15,AO15 = calcFeatures(img_edge,center_x,center_y,outerX[::10],outerY[::10])

    return NR,AR,AC15,AO15

#test code for this module
if __name__ == '__main__':

    #---csv---
    #usedata list
    model_file_path=r'C:\Users\sirim\Pictures\feature_extraction\learning_result'
    image_folder_path=r'C:\Users\sirim\Pictures\indoor\experiment_photo'

    GT_NR=  [41,34,38,23,45,39,36,42]
    GT_AR=  [0.253658537,0.219117647,0.325,0.339130435,0.244444444,0.242307692,0.227777778,0.283333333]
    GT_AC15=[0.452222222,0.316666667,0.4975,0.360833333,0.411111111,0.38,0.354444444,0.488333333]
    GT_AO15=[0.1,0.143333333,0.223333333,0.256666667,0.148333333,0.151111111,0.126666667,0.07]
    GT_AC15_mean = sum(GT_AC15)/len(GT_AC15)
    GT_AO15_mean = sum(GT_AO15)/len(GT_AO15)
    pxPerCm=[65.03,82.60,53,82.02,53,63.03,70.02,54]

    model_paths=[r'\100\before_otsu100.h5',
                r'\100\otsu100.h5',
                r'\100\before_th2_100.h5',
                r'\100\th2_100.h5',
                r'\1000\before_otsu1000.h5',
                r'\1000\otsu1000.h5',
                r'\1000\before_th2_1000.h5',
                r'\1000\th2_1000.h5']


    rootPath = pathlib.Path(image_folder_path)
    fp_list = list(rootPath.glob('*'))
    photo=0
    for fp in fp_list:

        if not (os.path.exists('result')):
            os.mkdir('result')
        path = 'result/{}.csv'.format(photo+1)
        if os.path.exists(path):
            os.remove(path)

        with open(path,'a',newline='') as c:
            writer = csv.writer(c)
            writer.writerow(['NR','AR','AC15','AO15',
                             'AC15_rmse','AO15_rmse'])

        with open(path,'a',newline='') as c:
                   writer = csv.writer(c)
                   writer.writerow([GT_NR[photo],GT_AR[photo],GT_AC15[photo],GT_AO15[photo],
                                     0,0])

        load_img_c = cv2.imread(str(fp))
        img_hsv = cv2.cvtColor(load_img_c, cv2.COLOR_BGR2HSV)
        _,_,load_img = cv2.split(img_hsv)

        blur = cv2.medianBlur(load_img,5)
        cimg = np.copy(cv2.cvtColor(load_img,cv2.COLOR_GRAY2BGR))
        circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,500,
                                param1=100,param2=50,minRadius=500,maxRadius=900)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        #cv2.namedWindow('detected circles', cv2.WINDOW_KEEPRATIO)
        #cv2.imshow('detected circles',cimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        NR,AR,AC15,AO15=extractByTraditional(load_img,i[0],i[1],i[2])
        with open(path,'a',newline='') as c:
            writer = csv.writer(c)
            writer.writerow([NR,AR/pxPerCm[photo],AC15/pxPerCm[photo],AO15/pxPerCm[photo],
                         np.sqrt((GT_AC15[photo]-(AC15/pxPerCm[photo]))**2),np.sqrt((GT_AO15[photo]-(AO15/pxPerCm[photo]))**2)])

        for mp in model_paths:
            model = keras.models.load_model(model_file_path+mp)
            if(('otsu' in mp) and ('before' not in mp)):
                thresh='otsu'
            if(('th2' in mp) and ('before' not in mp)):
                thresh='adaptive'
            if('before' in mp):
                thresh=None
            NR,AR,AC15,AO15=extractFeature(load_img,i[0],i[1],i[2],model,thresh)

            #write result
            with open(path,'a',newline='') as c:
                    writer = csv.writer(c)
                    writer.writerow([NR,AR/pxPerCm[photo],AC15/pxPerCm[photo],AO15/pxPerCm[photo],
                         np.sqrt((GT_AC15[photo]-(AC15/pxPerCm[photo]))**2),np.sqrt((GT_AO15[photo]-(AO15/pxPerCm[photo]))**2)])

        photo+=1

    sys.exit()



    #sequential extraction
    #50708,50716,B28604,B28616,B46404,B46408,B46412,B46808,B46816
    GT_NR=  [41,34,38,23,45,39,36,42]
    GT_AR=  [0.253658537,0.219117647,0.325,0.339130435,0.244444444,0.242307692,0.227777778,0.283333333]
    GT_AC15=[0.452222222,0.316666667,0.4975,0.360833333,0.411111111,0.38,0.354444444,0.488333333]
    GT_AO15=[0.1,0.143333333,0.223333333,0.256666667,0.148333333,0.151111111,0.126666667,0.07]
    GT_AC15_mean = sum(GT_AC15)/len(GT_AC15)
    GT_AO15_mean = sum(GT_AO15)/len(GT_AO15)
    pxPerCm=[65.03,82.60,53,82.02,53,63.03,70.02,54]

    EX_NR=[]
    EX_AR=[]
    EX_AC15=[]
    EX_AO15=[]

    #rootPath = pathlib.Path(r'E:\traning_data(murakami)\experiment_photo_folder')
    rootPath = pathlib.Path(r'C:\Users\sirim\Pictures\indoor\experiment_photo')
    fp_list = list(rootPath.glob('*'))
    photo=0
    for fp in fp_list:
        load_img_c = cv2.imread(str(fp))
        img_hsv = cv2.cvtColor(load_img_c, cv2.COLOR_BGR2HSV)
        _,_,load_img = cv2.split(img_hsv)

        #img_edge = obtainEdges(load_img,minVal=130,maxVal=130,filter_size=3)
        #cv2.namedWindow('img_edge', cv2.WINDOW_KEEPRATIO)
        #cv2.imshow("img_edge",img_edge)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        blur = cv2.medianBlur(load_img,5)
        cimg = np.copy(cv2.cvtColor(load_img,cv2.COLOR_GRAY2BGR))
        circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,500,
                                param1=100,param2=50,minRadius=500,maxRadius=900)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

        #cv2.namedWindow('detected circles', cv2.WINDOW_KEEPRATIO)
        #cv2.imshow('detected circles',cimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        NR,AR,AC15,AO15=extractByTraditional(load_img,i[0],i[1],i[2])

        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\otsu100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\before_otsu100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\otsu1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\before_otsu1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\th2_100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\before_th2_100.h5')
        #model = keras.models.load_model(r'C:\Users\sirim\Pictures\feature_extraction\learning_result\1000\otsu1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\before_th2_1000.h5')

        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\otsu100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\before_otsu100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\otsu1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\before_otsu1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\th2_100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\100\before_th2_100.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\th2_1000.h5')
        #model = keras.models.load_model(r'C:\Users\VIgpu01\Pictures\learning_result\1000\before_th2_1000.h5')
        #NR,AR,AC15,AO15=extractFeature(load_img,i[0],i[1],i[2],model,thresh='otsu')

        print("this image result")
        print("NR:{}".format(NR))
        print("GT_NR:{}".format(GT_NR[photo]))
        print("AR:{}cm".format(AR/pxPerCm[photo]))
        print("GT_AR:{}cm".format(GT_AR[photo]))
        print("AC15:{}cm".format(AC15/pxPerCm[photo]))
        print("GT_AC15:{}cm".format(GT_AC15[photo]))
        print("AO15:{}cm".format(AO15/pxPerCm[photo]))
        print("GT_AO15:{}cm".format(GT_AO15[photo]))

        EX_NR.append(NR)
        EX_AR.append(AR/pxPerCm[photo])
        EX_AC15.append(AC15/pxPerCm[photo])
        EX_AO15.append(AO15/pxPerCm[photo])
        photo+=1


    NR_RMSE=np.sqrt(np.mean((np.array(GT_NR)-np.array(EX_NR))**2))
    AR_RMSE=np.sqrt(np.mean((np.array(GT_AR)-np.array(EX_AR))**2))
    AC15_RMSE=np.sqrt(np.mean((np.array(GT_AC15)-np.array(EX_AC15))**2))
    AO15_RMSE=np.sqrt(np.mean((np.array(GT_AO15)-np.array(EX_AO15))**2))

    print("-result--------")
    print("NR_RMSE:{:.2f}".format(NR_RMSE))
    print("AR_RMSE:{:.2f}cm".format(AR_RMSE))
    print("AC15_RMSE:{:.2f}cm".format(AC15_RMSE))
    print("AO15_RMSE:{:.2f}cm".format(AO15_RMSE))
    print("AC15_ratio:{:.2f}%".format((AC15_RMSE/GT_AC15_mean)*100))
    print("AO15_ratio:{:.2f}%".format((AO15_RMSE/GT_AO15_mean)*100))

    sys.exit()



    #load_img_c = cv2.imread(r"C:\Users\sirim\Pictures\indoor_denoised_lm0p01\B46404.tif")
    #img_hsv = cv2.cvtColor(load_img_c, cv2.COLOR_BGR2HSV)
    #img_h,img_s,load_img = cv2.split(img_hsv)
    #load_img = cv2.imread(r"E:\traning_data(murakami)\49804.tif",0)
    #load_img = cv2.imread(r"E:\traning_data(murakami)\DSC_0573_g.tif",0)
    #cv2.namedWindow('load_img', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("load_img",load_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    #img_edge = obtainEdges(load_img,minVal=80,maxVal=80,filter_size=3)
    #cv2.namedWindow('img_edge', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow("img_edge",img_edge)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #sys.exit()

    #cv2.imwrite(r"C:\Users\sirim\Pictures\indoor_canny\50012.tif",img_edge)


    #load_img,172,185,308-160
    #blur = cv2.medianBlur(load_img,5)
    #cimg = obtainEdges(blur,minVal=50,maxVal=100,filter_size=3)
    #cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)
    #cv2.imshow('detected circles',cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #sys.exit()

    #49804.tif (blur,cv2.HOUGH_GRADIENT,1,500,param1=100,param2=50,minRadius=500,maxRadius=900)
    #cimg = np.copy(cv2.cvtColor(load_img,cv2.COLOR_GRAY2BGR))
    #circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,500,
    #                        param1=100,param2=50,minRadius=400,maxRadius=900)
    #circles = np.uint16(np.around(circles))
    #for i in circles[0,:]:
        # draw the outer circle
    #    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
    #    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    #print("width:{},height:{}".format(load_img.shape[1],load_img.shape[0]))
    #print("center:({},{})".format(i[0],i[1]))
    #print("radius:{}px".format(i[2]))
    #cv2.namedWindow('detected circles', cv2.WINDOW_KEEPRATIO)
    #cv2.imshow('detected circles',cimg)
    #cv2.imwrite(r'C:\Users\VIgpu01\Pictures\fe_result\detected_circles.tif',cimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #sys.exit()

    #start =time.time()
    #NR,AR,AC15,AO15=extractByTraditional(load_img,i[0],i[1],i[2])
    #elapsed_time =time.time()-start
    #print("elapsed_time:{}".format(elapsed_time)+"[sec]")

    #load_img = cv2.adaptiveThreshold(load_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,
    #                                 31,2)
    #cv2.imshow('load_img',load_img)
    #cv2.waitKey(0)

    #start =time.time()
    #model = keras.models.load_model(r'C:\Users\VIgpu01\workspace(murakami)\grid_search_result\result20201230_otsu_128px_5fold.h5')
    #NR,AR,AC15,AO15=extractFeature(load_img,i[0],i[1],i[2],model)
    #elapsed_time =time.time()-start
    #print("elapsed_time:{}".format(elapsed_time)+"[sec]")

   
