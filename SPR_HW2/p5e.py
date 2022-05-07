###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import cv2

if __name__ == '__main__':
    img = cv2.imread('leo2_mask.png')
    result = cv2.imread('result.png')
    error = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if result[i,j,0] != img[i,j,0]:
                error += 1
    print("error is :")
    print(error/(680*600))
