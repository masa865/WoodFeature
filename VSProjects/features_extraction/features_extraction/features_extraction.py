#for Features extraction of wood
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

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return x,y

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

#main function of this module
def extractFeature(img,centerX,senterY,radius,classList=None):

    #obtain edge
    img_edge = obtainEdges(img)

    #select Low-noise line 

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



