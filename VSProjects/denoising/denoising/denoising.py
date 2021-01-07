#Denoising for features extraction of wood
import cv2
import numpy as np
import sys
import random

#add noise function for making test image
def add_noise(I_t,MEAN = 0.0,STD = 20.0):

    #-----------------------------------------------------
    #I_t  :Image to add noise(1 channel gray scale)
    #MEAN :mean
    #STD  :standard deviation
    #-----------------------------------------------------

    #add gauss noise
    I_noise = I_t.astype(np.float64) + np.random.normal(MEAN,STD,I_t.shape)
    I_noise_clipped = (np.clip(I_noise, 0, 255)).astype(I_t.dtype) #specify the interval of [0,255]
    
    return I_noise_clipped

#Gauss Seidel solution to this problem
def Gauss_Seidel(u, d_x, d_y, b_x, b_y, f, MU, LAMBDA):
    [ROW_N,COL_N] = np.shape(f)

    U = np.vstack([u[1:ROW_N,:],np.reshape(u[ROW_N-1,:],[1,COL_N])]) + np.vstack([np.reshape(u[0,:],[1,COL_N]),u[0:ROW_N-1,:]]) \
       + np.hstack([u[:,1:COL_N],np.reshape(u[:,COL_N-1],[ROW_N,1])]) + np.hstack([np.reshape(u[:,0],[ROW_N,1]),u[:,0:COL_N-1]])
    D = np.vstack([np.reshape(d_x[0,:],[1,COL_N] ), d_x[0:ROW_N-1,:]]) - d_x \
       + np.hstack([np.reshape(d_y[:,0],[ROW_N,1] ), d_y[:,0:COL_N-1]]) - d_y
    B = -np.vstack([np.reshape(b_x[0,:],[1,COL_N] ), b_x[0:ROW_N-1,:]]) + b_x \
       - np.hstack([np.reshape(b_y[:,0],[ROW_N,1] ), b_y[:,0:COL_N-1]]) + b_y

    G = LAMBDA/(MU + 4*LAMBDA)*(U+D+B) + MU/(MU + 4*LAMBDA)*f #G means u_n+1
    return G
    

def shrink(x,y):
    t = np.abs(x) - y
    S = np.sign(x)*(t > 0) * t

    return S

#Total Variation (Split Bregman method)
def tvDenoiseSB(img,CYCLE=150,MU=0.1,LAMBDA=0.3,TOL=5):

    if(img.ndim != 2): #input error
        print('tvDenoiseSB() assumes input of 1-channel grayscale image.')
        sys.exit()

    #initialization
    [ROW_N,COL_N] = np.shape(img)
    u = img.astype(np.float64)
    d_x = np.zeros([ROW_N,COL_N])
    d_y = np.zeros([ROW_N,COL_N])
    b_x = np.zeros([ROW_N,COL_N])
    b_y = np.zeros([ROW_N,COL_N])

    print("[CYCLE,Err]")
    for cyc in range(CYCLE):
        u_n = Gauss_Seidel(u,d_x,d_y,b_x,b_y,img.astype(np.float64),MU,LAMBDA)
        Err = np.max(np.abs(u_n[2:ROW_N-2,2:COL_N-2] - u[2:ROW_N-2,2:COL_N-2]))
        if np.mod(cyc,10)==0:
            print([cyc,Err])
        if Err < TOL:
            break
        else:
            u = u_n
            nablax_u = np.vstack([u[1:ROW_N,:], np.reshape(u[ROW_N-1,:],[1,COL_N] )]) - u
            nablay_u = np.hstack([u[:,1:COL_N], np.reshape(u[:,COL_N-1],[ROW_N,1] )]) - u  
            d_x = shrink(nablax_u + b_x, 1/LAMBDA)
            d_y = shrink(nablay_u + b_y, 1/LAMBDA)
            b_x = b_x + (nablax_u - d_x)
            b_y = b_y + (nablay_u - d_y)

    reconstructed_img = u.astype(img.dtype)

    return reconstructed_img

##Total Variation for denoising(Zhang method)
def tvDenoise(img,lamda=0.3,timestep=0.01):

    if(img.ndim != 2): #input error
        print('tvDenoise() assumes input of 1-channel grayscale image.')
        sys.exit()

    nrow,ncol = img.shape
    I_temp = img

    for n in range(150):

        #Ix = img(2nd_col,...,end_col,end_col) - img(1st_col,1st_col,...,front of end_col)
        Ix = 0.5*(np.concatenate([I_temp[:,1:ncol],np.reshape(I_temp[:,ncol-1],(1,nrow)).T] ,axis=1)
                  - np.concatenate([np.reshape(I_temp[:,0],(1,nrow)).T ,I_temp[:,0:ncol-1]],axis=1))
        #Iy = img(2nd_row,...,end_row,end_row).T - img(1st_row,1st_row,...,front of end_row).T
        Iy = 0.5*(np.concatenate([I_temp[1:nrow,:],np.reshape(I_temp[nrow-1,:],(1,ncol))],axis=0)
                  - np.concatenate([np.reshape(I_temp[0,:],(1,ncol)),I_temp[0:nrow-1,:]],axis=0))

        Ix_back = I_temp - np.concatenate([np.reshape(I_temp[:,0],(1,nrow)).T ,I_temp[:,0:ncol-1]],axis=1)
        Ix_forw = np.concatenate([I_temp[:,1:ncol],np.reshape(I_temp[:,ncol-1],(1,nrow)).T] ,axis=1) - I_temp
        Iy_back = I_temp - np.concatenate([np.reshape(I_temp[0,:],(1,ncol)),I_temp[0:nrow-1,:]],axis=0)
        Iy_forw = np.concatenate([I_temp[1:nrow,:],np.reshape(I_temp[nrow-1,:],(1,ncol))],axis=0) - I_temp

        #grad = Ix.^2 + Iy.^2 + eps
        grad = np.power(Ix,2) + np.power(Iy,2) + np.spacing(1)

        Ixx = np.concatenate([I_temp[:,1:ncol],np.reshape(I_temp[:,ncol-1],(1,nrow)).T] ,axis=1)
        + np.concatenate([np.reshape(I_temp[:,0],(1,nrow)).T ,I_temp[:,0:ncol-1]],axis=1)
        - 2*I_temp
        Iyy = np.concatenate([I_temp[1:nrow,:],np.reshape(I_temp[nrow-1,:],(1,ncol))],axis=0)
        + np.concatenate([np.reshape(I_temp[0,:],(1,ncol)),I_temp[0:nrow-1,:]],axis=0)
        - 2*I_temp
        #Ixy = 
        #term1 =

        #D = gauss()
        #DD = 
        #upwind_x = 
        #upwind_x
        #upwind_y
        #upwind_y
        #term2
        #I_t

        #I_temp 





#test code for this module
if __name__ == '__main__':

    img_load = cv2.imread(r"C:\Users\sirim\Pictures\indoor image\no ruler\49804.tif")

    #plot original image
    #I_t = cv2.cvtColor(img_load, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("original",I_t)
    #cv2.waitKey(0)

    #plot noisy image
    #f = add_noise(I_t)
    #cv2.imshow("noisy",f)
    #cv2.waitKey(0)

    #plot reconstructed image
    #r = tvDenoiseSB(f)
    #cv2.imshow("reconstructed",r)
    #cv2.waitKey(0)

    #im_h = cv2.hconcat([cv2.hconcat([I_t, f]),r])
    #cv2.imshow("concat",im_h)
    #cv2.waitKey(0)

    img_hsv = cv2.cvtColor(img_load, cv2.COLOR_BGR2HSV)
    img_h,img_s,img_v = cv2.split(img_hsv)
    #cv2.imshow("img_v",img_v)
    #cv2.waitKey(0)

    img_r = tvDenoiseSB(img_v,CYCLE=50)

    #save image
    cv2.imwrite(r'C:\Users\sirim\Pictures\indoor_denoised\49804.tif',img_r)





    