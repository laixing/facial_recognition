import numpy as np
import pandas as pd
import os

root_dir = os.path.abspath('../Dataset')

def fer2013():

    data = pd.read_csv(os.path.join(root_dir,'new-mmi-video-CSV-diff-normalized-slim.csv'))
    data = np.asarray(data)
    np.random.shuffle(data) #must shuffle data to split correctly
    #Since size of dataset is 533, we are probably using kfold cv, so we split the set into trainval set and test set at 20% split
    train_y = data[0:int(data.shape[0]),0]
    train_y = train_y.astype(int)
    train_x = data[0:int(data.shape[0]),1:]

    train_y = onehot(train_y)

    train_x = scale(train_x)

    train_x = train_x.reshape((train_x.shape[0], 3, 48, 48))

    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    print(train_x.shape)
    print(train_y.shape)

    return (train_x, train_y)



def onehot(y):
    y_oh = np.zeros((len(y), np.max(y)+1))
    y_oh[np.arange(len(y)), y] = 1
    return y_oh


def scale(x):
    x = x.astype('float')
    x = x - 0.5
    X = np.zeros(shape=(x.shape[0],3,48*48))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if j%3 == 0: #r
                X[i,0,j/3] = x[i,j]
            elif j%3 ==1: #g
                X[i,1,j/3] = x[i,j]
            else: #b
                X[i,2,j/3] = x[i,j]
    x = x.reshape(-1, 3 * 48 * 48)
    return x
