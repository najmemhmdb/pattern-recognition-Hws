###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import cv2
from scipy.fftpack import dct
import numpy as np


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


if __name__ == '__main__':
    test_img = cv2.imread('leo2.png')
    zigzag = np.loadtxt("zigzag_pattern.txt")
    output = []
    for i in range(int(test_img.shape[0]/8)):
        for j in range(int(test_img.shape[1]/8)):
            block = test_img[i*8:(i+1)*8,j*8:(j+1)*8]
            D = dct2(block[..., 2])
            vector = np.empty(64, dtype=float)
            index = zigzag[0, 0]
            for row in range(8):
                for column in range(8):
                    index = zigzag[row,column]
                    vector[int(index)] = int(abs(D[row,column]))
            vector.sort()
            x = vector[62]
            output.append(x)
    output = np.array(output)
    a_file = open("test_x.txt", "w")
    np.savetxt(a_file, output)
    a_file.close()