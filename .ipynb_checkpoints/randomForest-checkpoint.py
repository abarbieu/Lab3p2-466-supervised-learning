#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import json
import sys
import time
from InduceC45 import c45, readFiles
from classifier import classify, evaluate, traverseTree


# In[2]:


def sampleDf(df, attrs, m, k):
    datapoints = df.sample(frac=k)
    newdata = datapoints.iloc[:,:-1].sample(m,axis=1).join(datapoints.iloc[:,-1])
    
    newattrs = {x: attrs[x] for x in newdata.iloc[:,:-1].columns}
    return newdata, newattrs


# In[3]:


def createRandomForest(df, attrs, m, k, n, thresh=0.2):
    trees=[]
    for i in range(n):
        sample, sampleAttrs = sampleDf(df,attrs,m,k)
        trees.append(c45(sample, sampleAttrs, thresh))
    return trees


# In[4]:


def classifyForest(df, forest, asList=False, getAccuracy=False):
    preds = []
    for i, row in df.iterrows():
        predsmini = []
        for tree in forest:
            keys = list(tree)
            predsmini.append(traverseTree(row, tree[keys[-1]], keys[-1]))
        preds.append([i, pd.Series(predsmini).mode()[0]])
    
    preddf=None
    accuracy=None
    
    if getAccuracy or not asList:
        preddf = pd.DataFrame(preds, columns=['index', 'prediction']).set_index('index')
    
    if getAccuracy:
        numCorrect=0
        numClassified=0
        for i, row in df.iterrows():
            if preddf.loc[i,"prediction"] == row[df.columns[-1]]:
                numCorrect += 1
            numClassified += 1
            
        accuracy = numCorrect/numClassified
    
    if asList:
        return preds, accuracy
    else:
        return preddf, accuracy


# In[14]:


def crossValidateForest(df, attrs, numSplits, m, k, n, thresh=0.2):
    prev=None
    kfoldPreds = []
    cumOutput = None
    accuracies = []
    # all but one cross validation
    if numSplits == -1:
        numSplits = len(df)-1
    
    # split dataset kfold and generate predictions
    if numSplits <= 1:
        res, acc= classifyForest(df, createRandomForest(df, attrs, m, k, n, thresh), asList=True, getAccuracy=True)
        kfoldPreds += res
        accuracies.append(acc)
    else:
        splitnum=0
        # go through indecies by fold length
        for i in range(0, len(df), int(len(df)/numSplits)):
            splitnum+=1
            if prev is None:
                prev=i
            else:
                trainingData = pd.concat([df[:prev], df[i:]])
                classifyData = df[prev:i]
                forest=createRandomForest(trainingData, attrs, m, k, n, thresh)
                
                res, acc= classifyForest(classifyData, forest, asList=True, getAccuracy=True)
                kfoldPreds += res
                accuracies.append(acc)
                prev=i
        
        trainingData = df[:prev]
        classifyData = df[prev:]
        res, acc= classifyForest(classifyData, createRandomForest(trainingData, attrs, m, k, n, thresh), asList=True, getAccuracy=True)
        kfoldPreds += res
        accuracies.append(acc)
    
#     print("split accuracies:", accuracies, np.sum(accuracies))
    results = evaluate(df, kfoldPreds, asList=True)
    
    print("Average Accuracy:", np.sum(accuracies)/numSplits)
    for v in results:
        print(v, ":\n", results[v])
    
    return df.join(pd.DataFrame(kfoldPreds, columns=['index', 'prediction']).set_index('index'))


# In[16]:


if __name__ == "__main__":
    thresh=0.2
    if len(sys.argv) == 6:
        _, datafile, m, k, n, outputfile = sys.argv
    elif len(sys.argv) == 7:
        _, datafile, m, k, n, outputfile, thresh = sys.argv
    else:
        print("Usage: python3 randomForest.py <datafile.csv> <m: numAttrs>  <k: percentDataPoints> <N: numTrees> <outputFileName.csv> [thresh=0.2]")
        exit(1)
    
    m=int(m)
    k=float(k)
    n=int(n)
    thresh=float(thresh)
#     m = 2
#     k = .2
#     n = 10
#     datafile = "./data/iris.data.csv"
#     outputfile = "./output/iris.forest.preds.csv"
    
    df, filename, isLabeled, attrs = readFiles(datafile)
    
    outDf = crossValidateForest(df, attrs, 10, m, k, n, thresh)
    
    outDf.to_csv(outputfile)
 






