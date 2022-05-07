###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import cv2
from scipy.fftpack import dct
import numpy as np
import matplotlib.pyplot as plt

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

if __name__ == '__main__':
    prior_messi = 0
    prior_background = 0
    x_conditional_messi = []
    x_conditional_background = []
    zigzag = np.loadtxt("zigzag_pattern.txt")
    img = cv2.imread('leo1.png')
    guide = cv2.imread('leo1_mask.png')
    for i in range(int(img.shape[0]/8)):
        for j in range(int(img.shape[1]/8)):
            black = 0
            white = 0
            block = img[i*8:(i+1)*8,j*8:(j+1)*8]
            guide_block = guide[i*8:(i+1)*8,j*8:(j+1)*8]
            D = dct2(block[..., 2])
            vector = np.empty(64, dtype=float)
            index = zigzag[0, 0]
            for row in range(8):
                for column in range(8):
                    index = zigzag[row,column]
                    vector[int(index)] = int(abs(D[row,column]))
                    if guide_block[row,column,0] == 0:
                        prior_background+=1
                        black += 0
                    else:
                        prior_messi+= 1
                        white += 1
            vector.sort()
            x = vector[62]
            if black >= white:
                x_conditional_background.append(x)
            else:
                x_conditional_messi.append(x)
    prior_messi /= 408000
    prior_background /= 408000
    x_conditional_messi = np.array(x_conditional_messi)
    x_conditional_background = np.array(x_conditional_background)
    a_file = open("x_messi.txt", "w")
    np.savetxt(a_file, x_conditional_messi)
    a_file.close()
    a_file = open("x_background.txt","w")
    np.savetxt(a_file,x_conditional_background)
    a_file.close()
    a_file = open("prior_probability.txt","w")
    np.savetxt(a_file,np.array([prior_messi,prior_background]))
    a_file.close()
    fig, axs = plt.subplots(1,2, sharey=True, tight_layout=True)
    a,b,c = axs[0].hist(x_conditional_messi, bins=int(max(x_conditional_messi)))
    axs[0].set_xlabel("messi")
    a1,b1,c1 = axs[1].hist(x_conditional_background, bins=int(max(x_conditional_background)))
    axs[1].set_xlabel("field")
    plt.show()
