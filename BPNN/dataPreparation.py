#coding:utf-8


import numpy as np


SEED=2

def generateDataSets():
    
    rng=np.random.RandomState(SEED)
    X_trainData=rng.randn(300,2)

    Y_trainData=[int(x0*x0+x1*x1<1.5) for (x0,x1) in X_trainData]
    
    Y_trainData=np.vstack(Y_trainData)
    
    Y_color=[["red" if y else "blue"] for y in Y_trainData]

    print("X_trainData:",X_trainData)
    print("Y_trainData:",Y_trainData)
    print("Y_color:",Y_color)
    
    return X_trainData, Y_trainData,Y_color

#generateDataSets()
