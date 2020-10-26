#Preprocessing for features extraction of wood
import cv2

def preprocess(img_bgr):
    #extract V of HSV space
    img_v = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    
    return img_v

#test code
if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\sirim\Pictures\new\deru.jpg")
    prepro_img = preprocess(img)
    print("shape")
    print(prepro_img.shape)
    print("data type")
    print(prepro_img.dtype)
    cv2.imshow('image',prepro_img)
    cv2.waitKey(0)