import sklearn .svm
import numpy as np

from dataset_greyscale import mmi_vid_greyscale
from PCA import PCA_dim_red
from dataset import mmi_vid
import numpy as np
from Landmarks import Landmarks
import scipy
import os
from HOG import HOG
import pickle
from HOG_and_Landmarks import Hog_and_Landmarks
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
print("Importing data")
(train_x, train_y), (test_x, test_y)= mmi_vid_greyscale()

# Hog and landmarks
#train_x, test_x=PCA_dim_red(train_x,test_x,0.99)
print(np.shape(train_x))
print(np.shape(test_x))
train_x, test_x=Hog_and_Landmarks(train_x,test_x)
# train_x, test_x=PCA_dim_red(train_x,test_x,0.99)
print("Imported data, initiating training")

train_y = np.ravel(train_y)
test_y = np.ravel(test_y)
k=10
kf = KFold(n_splits=k)

vals_y = []
Poly_preds_y = []
Gaussian_preds_y = []

SVM1 = sklearn.svm.SVC(C=25.0, kernel='poly', coef0=1.0)
SVM2 = sklearn.svm.SVC(C=25.0, kernel='rbf')

errork = np.zeros(k)
errork1= np.zeros(k)
counter=0
for train_idx, val_idx in kf.split(train_x):
    X_train, X_val = train_x[train_idx], train_x[val_idx]
    y_train, y_val = train_y[train_idx], train_y[val_idx]

    SVM1.fit(X_train, y_train)
    SVM2.fit(X_train, y_train)
    Poly_pred_y = SVM1.predict(X_val)
    #print("prediction:")
    #print(Poly_pred_y)
    accuracy = np.mean(Poly_pred_y==y_val)
    errork[counter] = 1-accuracy

    Gaussian_pred_y = SVM2.predict(X_val)
    accuracy1=np.mean(Gaussian_pred_y==y_val)
    errork1[counter]=1-accuracy1

    counter+=1

error10fold = np.mean(errork)
error10fold1=np.mean(errork1)
print(vals_y)
print(Poly_pred_y)
print(Gaussian_pred_y)
print('---------------------------- SVM Polynomial----------------------------------------------')

print("SVM validation error : " + str(error10fold))

print('---------------------------- SVM Gauusian-------------------------------------------------')

print("SVM validation error : " + str(error10fold1))

preds_ty1 = np.ravel(SVM1.predict(test_x))
test_accuracy = np.mean(preds_ty1 == test_y)
print("SVM test accuracy for poly : " + str(test_accuracy))
conmatrix=confusion_matrix(test_y, preds_ty1)
print(conmatrix)
filename = 'svm_poly.sav'
pickle.dump(SVM1, open(filename, 'wb'))

preds_ty = np.ravel(SVM2.predict(test_x))
test_accuracy = np.mean(preds_ty == test_y)
print("SVM test accuracy for Gasussian : " + str(test_accuracy))
