#module for finding the center of annual rings
import cv2
import numpy as np
import time

#Hough Circle Transform(with OpenCV function)
def hough(img,dp=1,minDist=1500,param1=130,param2=130):

    #1_8 =>1sec     dp=1,minDist=800,param1=80,param2=50
    #1_4 =>1sec     dp=1,minDist=800,param1=200,param2=150 *circle can't detect
    #1_2 =>4.356sec dp=1,minDist=1500,p1=550,p2=220 *circle can't detect

    h,w = img.shape
    maxr = np.min(img.shape)
    minr = maxr // 3

    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp,minDist,
                                param1,param2,minRadius=minr,maxRadius=maxr)

    circles = np.uint16(np.around(circles))
    
    return circles #circles = [centerX,centerY,radius]


#test code for this module
if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\sirim\Pictures\new\output\49808_r.tif",0)
   

    start = time.time()
    circles = hough(img)
    elapsed_time = time.time() - start
    print ("hough() time:{:.3f}".format(elapsed_time) + "[sec]")
    
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,0,255),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.imwrite(r"C:\Users\sirim\Pictures\new\output\49808_r_cent.tif",cimg)
    cv2.destroyAllWindows()
