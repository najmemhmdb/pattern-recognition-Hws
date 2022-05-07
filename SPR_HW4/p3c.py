###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import  numpy as np
import pandas as pd

def calculate_scatter(x):
    mean = x[['PressureLoad', 'DefVol' , 'Bacteria' , 'Yeast', 'pH', 'TTA']].mean()
    for row in range(len(x)):
        x.iloc[row]['PressureLoad'] = x.iloc[row]['PressureLoad'] - mean['PressureLoad']
        x.iloc[row]['DefVol'] = x.iloc[row]['DefVol'] - mean['DefVol']
        x.iloc[row]['Bacteria'] = x.iloc[row]['Bacteria'] - mean['Bacteria']
        x.iloc[row]['Yeast'] = x.iloc[row]['Yeast'] - mean['Yeast']
        x.iloc[row]['pH'] = x.iloc[row]['pH'] - mean['pH']
        x.iloc[row]['TTA'] = x.iloc[row]['TTA'] - mean['TTA']
    X = x.to_numpy()
    return np.dot(X[:,0:6].T,X[:,0:6])

if __name__ == '__main__':
    data = pd.read_table("doughs.dat",sep=" ")
    data = data.astype('float64')
    S = calculate_scatter(data)
    eig_val , eig_vec = np.linalg.eig(S)
    A = [eig_vec[0],eig_vec[1],eig_vec[2]]
    A = np.array(A)
    X = np.array(data.iloc[:,0:6])
    Y = np.dot(X,A.T)
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(Y.shape)
    print(Y[0:5,0], Y[0:5,1], Y[0:5,2])
    ax.scatter(Y[0:20,0], Y[0:20,1], Y[0:20,2], marker="*", color="red",label="Naples")
    ax.scatter(Y[20:30,0], Y[20:30,1], Y[20:30,2], marker="*" , color="green",label="others")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.legend()

    # 2D plot
    # plt.scatter(Y[0:20,1], Y[0:20,2], marker="*", color="red",label="Naples")
    # plt.scatter(Y[20:30,1], Y[20:30,2], marker="*" , color="green",label="others")
    # plt.legend()
    # plt.title('second and third PCs')
    plt.show()



#