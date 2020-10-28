#Preprocessing for features extraction of wood
import cv2

def preprocess(img_bgr):
    #extract V of HSV space
    img_v = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    img_prep = img_v
    
    return img_prep


#test code for this module
if __name__ == '__main__':
    img = cv2.imread(r"C:\Users\sirim\Pictures\new\deru.jpg")
    img_prep = preprocess(img)
    print("shape")
    print(img_prep.shape)
    print("data type")
    print(img_prep.dtype)
    cv2.imshow('image',img_prep)
    cv2.waitKey(0)