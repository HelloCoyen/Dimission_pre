# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:04:23 2020

@author: wuhaoyu
"""

from numpy import *
import operator

#def createDataSet():
#    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#    labels = ['A','A','B','B']
#    return group, labels
#
#dataSet = createDataSet()[0]
#labels = createDataSet()[1]
#inX = [0,0]
#k = 2

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]
