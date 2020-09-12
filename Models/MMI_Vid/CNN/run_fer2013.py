"""
Noting that this is a different dataset (MMI vid) from Kaggle, but size of input is the same (Kaggle was 48x48, MMI Vid is 48x48)
"""
from layers.conv import ConvLayer
from layers.flatten import FlattenLayer
from layers.maxpool import MaxPoolLayer
from layers.relu import ReluLayer
from layers.sequential import Sequential
from layers.softmax import SoftMaxLayer
from layers.full import FullLayer
from layers.cross_entropy import CrossEntropyLayer
from layers.dataset import fer2013
import numpy as np
from sklearn.model_selection import KFold

print('Importing data')
myX, y = fer2013()

#5 Fold cross validation
k = 3
kf = KFold(n_splits=k)

lr = 0.3
epochs = 60
batch_size = 32

counter = 0
errork = np.zeros(k)
loss = np.zeros(shape=(k,epochs))
for train_index, test_index in kf.split(myX[np.arange(533),:,:,:]):
    train_x, test_x = myX[train_index,:,:,:],myX[test_index,:,:,:]
    train_y, test_y = y[train_index],y[test_index]
    #training
    print('Creating model with lr = ' + str(lr))
    myNet = Sequential(layers=(ConvLayer(n_i=3, n_o=16, h=3),
                               ReluLayer(),
                               MaxPoolLayer(size=2),
                               ConvLayer(n_i=16, n_o=32, h=3),
                               ReluLayer(),
                               MaxPoolLayer(size=2),
                               FlattenLayer(),
                               FullLayer(n_i=12 * 12 * 32, n_o=6),  # no neutral class:/
                               SoftMaxLayer()),
                       loss=CrossEntropyLayer())

    print("Initiating training")
    loss[counter,:] = myNet.fit(x=train_x, y=train_y, epochs=epochs, lr=lr, batch_size=batch_size)
    myNet.save()
    pred = myNet.predict(test_x)
    accuracy = np.mean(pred == test_y)
    errork[counter] = 1 - accuracy
    print('At fold = '+str(counter+1))
    print('Accuracy of Convolutional Neural Network = ' + str(accuracy))
    counter += 1
error5folds = np.mean(errork)
accuracy = 1 - error5folds
print('At learning rate = ' + str(lr)+ 'and fold = '+str(counter+1))
print('Accuracy of Convolutional Neural Network = ' + str(accuracy))



 
