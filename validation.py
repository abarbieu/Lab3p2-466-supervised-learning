#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import json
import sys
from InduceC45 import c45, readFiles
from classifier import classify, readArrange, initializeConfusion

# In[85]:
def getArgs():
    restr=None
    if len(sys.argv) < 3:
        print("Not enough arguments.")
        exit(1)
    elif len(sys.argv) == 3:
        file1 = sys.argv[1]
        k = sys.argv[2]
    elif len(sys.argv) == 4:
        file1 = sys.argv[1]
        k = sys.argv[2]
        restr = sys.argc[3]
    
    df,tmp,isLabeled = readFiles(file1,restr)
    
    return df, int(k), isLabeled

def predict_kfold(df, numSplits, threshold, isLabeled):
    prev=None
    kfoldPreds = []
    accCorr = [0, 0]
    confusion = initializeConfusion(df)
    
    # all but one cross validation
    if numSplits == -1:
        numSplits = len(df)-1
    
    # split dataset kfold and generate predictions
    if numSplits <= 1:
        kfoldPreds += classify(accCorr, confusion, df, c45(df, df.columns[:-1].tolist(), threshold), silent=True, 
                               labeled=isLabeled)
    else:
        splitnum=0
        # go through indecies by fold length
        for i in range(0,len(df),int(len(df)/numSplits)):
            splitnum+=1
            if prev is None:
                prev=i
            else:
                trainingData = pd.concat([df[:prev], df[i:]])
                classifyData = df[prev:i]
                tree=c45(trainingData, df.columns[:-1].tolist(), threshold)
                kfoldPreds += classify(accCorr, confusion, classifyData, tree, silent=True, labeled=isLabeled)
                prev=i
        
        trainingData = df[:prev]
        classifyData = df[prev:]
        kfoldPreds += classify(accCorr, confusion, classifyData, c45(trainingData, df.columns[:-1].tolist(), threshold), silent=True, 
                               labeled=isLabeled)
    
    ret = pd.DataFrame(kfoldPreds, columns=['index', 'prediction']).set_index('index')
    ret['actual'] = df.loc[:,df.columns[-1]:]
    
    print()
    print(f"-----Ran {numSplits}-fold cross-validation-----")
    print("Overall Accuracy: ", accCorr[1]/len(ret))
    print("Average Accuracy: ", accCorr[0]/numSplits)
    print("Confusion Matrix:\n", confusion)
    return ret


if __name__ == '__main__':
    df, k, isLabeled = getArgs()
    print(predict_kfold(df, k, 0.2, isLabeled))