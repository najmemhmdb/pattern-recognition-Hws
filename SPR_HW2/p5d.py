###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_conditional_messi = np.loadtxt("x_messi.txt")
    x_conditional_background = np.loadtxt("x_background.txt")
    messi_prior , bg_prior = np.loadtxt("prior_probability.txt")
    test_x = np.loadtxt("test_x.txt")
    A,B,C = plt.hist(x_conditional_messi,bins=int(max(x_conditional_messi)))
    A2,B2,C2 = plt.hist(x_conditional_background,bins=int(max(x_conditional_background)-1))
    label = []
    for i in range(test_x.size):
        if test_x[i] <= 124 and test_x[i]>= 1 :
            messi = A[int(test_x[i])]
            background = A2[int(test_x[i])]
        elif test_x[i] >= 125 or test_x[i] == 0  :
            messi = 1
            background = 0
        if messi*messi_prior > background*bg_prior:
            label.append(255)
        else:
            label.append(0)
    label = np.array(label)
    a_file = open("label.txt", "w")
    np.savetxt(a_file, label)
    a_file.close()
    img = cv2.imread('leo2.png')
    counter = 0
    for i in range(int(img.shape[0]/8)):
        for j in range(int(img.shape[1]/8)):
            for row in range(8):
                for column in range(8):
                    img[i*8+row,j*8+column,0] = label[counter]
                    img[i*8+row,j*8+column,1] = label[counter]
                    img[i*8+row,j*8+column,2] = label[counter]
            counter += 1
    # cv2.imshow('image',img)
    cv2.imwrite("result.png", img)
    # cv2.waitKey()
    #