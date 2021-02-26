import tensorflow as tf
import numpy as np
import os
import scipy.io
from sklearn.model_selection import train_test_split, KFold
import hdf5storage
import pylatex as lat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###################################### fake data creation ##############################################################

feature_size = 12

def FakeData():
    sample_size = 1000
    pos_feature = [1,3]     # pos_feat means : if higher value, higher proba of being defaulted
    neg_feature = [2]       # neg_feat means : if higher value, lower proba of being defaulted
    x = np.random.normal(size=(sample_size,feature_size))

    temp=np.add(np.sum(x[:,pos_feature].clip(0,100),axis=1),
                np.abs(x[:,neg_feature].clip(-100,0)).reshape(sample_size))
    percentage_default = 0.05
    treshold=np.quantile(temp,1-percentage_default)
    noise_epsilon=0.1
    y = 1*((temp+np.random.normal(size=sample_size,scale=noise_epsilon))>treshold)
    return x,y

###################################### duan data creation ##############################################################

path = os.path.join('C:\\Users', 'divernoi', 'Dropbox', 'PhD', 'Duan Replication')

# firm0 : surviving, firm1 : default, firm2 : otherexits
test_cut = 0.2

def standardize_data(inx):
    stdx = np.std(inx[:, 1:13], axis=0)
    meanx = np.mean(inx[:, 1:13], axis=0)
    outx = (inx[:, 1:13] - np.mean(inx[:, 1:13], axis=0)) / np.std(inx[:, 1:13], axis=0)
    return stdx, meanx, outx


def RealData_f(tau):
    fulldata = hdf5storage.loadmat(os.path.join(path, 'fulldata.mat'))
    numpdata = np.array(list(fulldata.items()))[0][1][tau]
    x0 =  np.concatenate((numpdata[0],  numpdata[2]), axis=0)                                 # select the 0 indices for likelihood term 0
    y0 = np.zeros(numpdata[0].shape[0] + numpdata[2].shape[0])
    x1 = numpdata[1]                                                                          # select the 1 indices for likelihood term 1
    y1 = np.ones(numpdata[1] .shape[0])
    x_train0, x_test0, y_train0, y_test0 = train_test_split(x0, y0, test_size=test_cut)         # split dataset 0 into train and test set
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=test_cut)         # split dataset 1 into train and test set ... need to make it in 2 steps to be sure to have enough data of both indices
    x_train = np.concatenate((x_train0,x_train1),axis=0)
    y_train = np.concatenate((y_train0, y_train1), axis=0)
    x_test = np.concatenate((x_test0, x_test1), axis=0)
    y_test = np.concatenate((y_test0, y_test1), axis=0)
    train_std, train_mean, x_train = standardize_data(x_train)                                  # store std,mean of train set and standardize train set
    x_test = (x_test[:, 1:13] - train_mean) / train_std                                         # standardize test set using mean,std of train set
    return x_train,y_train,x_test,y_test, train_mean,train_std

def RealData_h(tau):
    fulldata = hdf5storage.loadmat(os.path.join(path, 'fulldata.mat'))
    numpdata = np.array(list(fulldata.items()))[0][1][tau]
    x0 = numpdata[0]
    y0 = np.zeros(numpdata[0].shape[0])
    x1 = numpdata[2]
    y1 = np.ones(numpdata[2].shape[0])
    x_train0, x_test0, y_train0, y_test0 = train_test_split(x0, y0, test_size=test_cut)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=test_cut)
    x_train = np.concatenate((x_train0,x_train1),axis=0)
    y_train = np.concatenate((y_train0, y_train1), axis=0)
    x_test = np.concatenate((x_test0, x_test1), axis=0)
    y_test = np.concatenate((y_test0, y_test1), axis=0)
    train_std, train_mean, x_train = standardize_data(x_train)
    x_test = (x_test[:, 1:13] - train_mean) / train_std
    return x_train,y_train,x_test,y_test, train_mean,train_std

#################################### k fold validation #################################################################

# data sample
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
y = np.array([0,0,0,0,1,1])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(x):
	print('xtrain: %s, xtest: %s , ytrain: %s , ytest: %s' % (x[train], x[test] , y[train] , y[test]))

N = []

#for tau in range(36):
#    fulldata = hdf5storage.loadmat(os.path.join(path, 'fulldata.mat'))
#    numpdata = np.array(list(fulldata.items()))[0][1][tau]
#    N.append((numpdata[0].shape[0],numpdata[1].shape[0],numpdata[2].shape[0]))

