###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
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
    # print(data)
    S = calculate_scatter(data)
    eig_val , eig_vec = np.linalg.eig(S)
    A = [eig_vec[0],eig_vec[1],eig_vec[2]]
    A = np.array(A)
    X = np.array(data.iloc[:,0:6])
    Y = np.dot(X,A.T)
    sigma = 0
    sigma_2 = 0
    for x in X:
        sigma += np.dot(x,x.T)
    for a in A:
        mag = np.linalg.norm(a)
        e = a / mag
        ets = np.dot(e.T,S)
        sigma_2 += np.dot(ets,e)
    print("representation error:")
    print(sigma - sigma_2)
    for i in eig_val:
        print(i)


