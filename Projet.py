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

import matplotlib.pyplot as plt
import matplotlib.image as im

x = pd.read_csv('./50signDataset/GT-00002.csv', sep='\;')
y = pd.read_csv('./30signDataset/GT-00001.csv', sep=';')
w = pd.read_csv('./60signDataset/GT-00003.csv', sep=';')

nameOne = y.Filename
is30Sign = y.ClassId

nameTwo = x.Filename
is50Sign = x.ClassId

nameThree = w.Filename
is60Sign = w.ClassId


tabIThirty = np.zeros([2220,55*57])
tabClassIdThirty = np.zeros([2220,1])

for i in range(1,2220) :
    I=im.imread("30signDataset/" + nameOne[i])
    tabClassIdThirty[i] = is30Sign[i]
    
    IgrayOne=cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    IgrayOne=cv2.resize(IgrayOne,(57,55),interpolation = cv2.INTER_AREA)

    tabIThirty[i,:] = IgrayOne.reshape(IgrayOne.shape[0]*IgrayOne.shape[1])
    
K=tabIThirty[1,:].reshape([55,57])
plt.imshow(K)

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

M=tabISixty[1,:].reshape([55,57])
plt.imshow(M)

tabIThirty = pd.DataFrame(tabIThirty)
tabClassIdThirty = pd.DataFrame(tabClassIdThirty)

print(type(tabIThirty))
print(type(tabClassIdThirty))

final30SignTab = tabIThirty

tabIFifty = pd.DataFrame(tabIFifty)
tabClassIdFifty = pd.DataFrame(tabClassIdFifty)

print(tabIFifty.shape)
print(tabIThirty.shape)

final50SignTab = tabIFifty

finalClassIdTab = pd.concat([tabClassIdThirty,tabClassIdFifty])

print(final50SignTab.shape)
print(finalClassIdTab.shape)
#print(type(final50SignTab))

finalTotalTab = pd.concat([final50SignTab,final30SignTab], ignore_index=True)
finalTotalTab.shape

xtrain, xtest, ytrain, ytest = train_test_split(finalTotalTab, finalClassIdTab, test_size=0.35, random_state=42)

print(xtrain.shape)
print(ytrain.shape)
print('pourcentage:' ,xtrain.shape[0]/finalTotalTab.shape[0])


Arbre_decision = DecisionTreeClassifier(random_state=0, max_depth=20)

clf = Arbre_decision.fit(xtrain, ytrain)


ypredit = clf.predict(xtest)
accuracy_score(ytest, ypredit)


print(metrics.confusion_matrix(ytest, ypredit))

KNN = KNeighborsClassifier()
clf = KNN .fit(xtrain, ytrain)
ypredit = clf.predict(xtest)
accuracy_score(ytest, ypredit)
print(metrics.confusion_matrix(ytest, ypredit))
