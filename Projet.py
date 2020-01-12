import pandas as pd

import numpy as np
from numpy import array

import cv2 as cv2

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


import matplotlib.pyplot as plt
import matplotlib.image as im

x = pd.read_csv('./50signDataset/GT-00002.csv', sep='\;')
y = pd.read_csv('./30signDataset/GT-00001.csv', sep=';')
w = pd.read_csv('./60signDataset/GT-00003.csv', sep=';')
z = pd.read_csv('./70signDataset/GT-00004.csv', sep=';')
v = pd.read_csv('./80signDataset/GT-00005.csv', sep=';')

nameOne = y.Filename
is30Sign = y.ClassId

nameTwo = x.Filename
is50Sign = x.ClassId

nameThree = w.Filename
is60Sign = w.ClassId

nameFour = z.Filename
is70Sign = z.ClassId

nameFive = v.Filename
is80Sign = v.ClassId

tabIThirty = np.zeros([2220,55*57])
tabClassIdThirty = np.zeros([2220,1])

for i in range(1,2220) :
    I=im.imread("30signDataset/" + nameOne[i])
    tabClassIdThirty[i] = is30Sign[i]
    
    IgrayOne=cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    IgrayOne=cv2.resize(IgrayOne,(57,55),interpolation = cv2.INTER_AREA)

    tabIThirty[i,:] = IgrayOne.reshape(IgrayOne.shape[0]*IgrayOne.shape[1])
    
tabIFifty = np.zeros([2220,55*57])
tabClassIdFifty= np.zeros([2220,1])

for i in range(1,2220) :
    I=im.imread("50signDataset/" + nameTwo[i])
    tabClassIdFifty[i] = is50Sign[i]
    
    IgrayTwo=cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    IgrayTwo=cv2.resize(IgrayTwo,(57,55),interpolation = cv2.INTER_AREA)

    tabIFifty[i,:] = IgrayTwo.reshape(IgrayTwo.shape[0]*IgrayTwo.shape[1])
    
tabISixty = np.zeros([1410,55*57])
tabClassIdSixty= np.zeros([1410,1])

for i in range(1,1410) :
    M=im.imread("60signDataset/" + nameThree[i])
    tabClassIdSixty[i] = is60Sign[i]
    
    IgrayThree=cv2.cvtColor(M, cv2.COLOR_BGR2GRAY)
    IgrayThree=cv2.resize(IgrayThree,(57,55),interpolation = cv2.INTER_AREA)

    tabISixty[i,:] = IgrayThree.reshape(IgrayThree.shape[0]*IgrayThree.shape[1])
    
tabISeventy = np.zeros([1410,55*57])
tabClassIdSeventy= np.zeros([1410,1])

for i in range(1,1410) :
    N=im.imread("70signDataset/" + nameFour[i])
    tabClassIdSeventy[i] = is70Sign[i]
    
    IgrayFour=cv2.cvtColor(N, cv2.COLOR_BGR2GRAY)
    IgrayFour=cv2.resize(IgrayFour,(57,55),interpolation = cv2.INTER_AREA)

    tabISeventy[i,:] = IgrayFour.reshape(IgrayFour.shape[0]*IgrayFour.shape[1])
    
tabIEighty = np.zeros([1860,55*57])
tabClassIdEigthy= np.zeros([1860,1])

for i in range(1,1860) :
    O=im.imread("80signDataset/" + nameFive[i])
    tabClassIdEigthy[i] = is80Sign[i]
    
    IgrayFive=cv2.cvtColor(O, cv2.COLOR_BGR2GRAY)
    IgrayFive=cv2.resize(IgrayFive,(57,55),interpolation = cv2.INTER_AREA)

    tabIEighty[i,:] = IgrayFive.reshape(IgrayFive.shape[0]*IgrayFive.shape[1])
    
K=tabIThirty[1,:].reshape([55,57])

L=tabIFifty[1,:].reshape([55,57])

M=tabISixty[1,:].reshape([55,57])

N=tabIFifty[1,:].reshape([55,57])

O=tabISixty[1,:].reshape([55,57])

plt.imshow(K)

final30SignTab = pd.DataFrame(tabIThirty)
tabClassIdThirty = pd.DataFrame(tabClassIdThirty)

final50SignTab = pd.DataFrame(tabIFifty)
tabClassIdFifty = pd.DataFrame(tabClassIdFifty)

final60SignTab = pd.DataFrame(tabISixty)
tabClassIdSixty = pd.DataFrame(tabClassIdSixty)

final70SignTab = pd.DataFrame(tabISeventy)
tabClassIdSeventy = pd.DataFrame(tabClassIdSeventy)

final80SignTab = pd.DataFrame(tabIEighty)
tabClassIdEigthy = pd.DataFrame(tabClassIdEigthy)

finalClassIdTab = pd.concat([tabClassIdThirty,tabClassIdFifty,tabClassIdSixty,tabClassIdSeventy,tabClassIdEigthy])
