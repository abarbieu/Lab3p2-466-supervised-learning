#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import json
import sys
from InduceC45 import c45, readFiles
from classifier import classify, evaluate, initializeConfusion

# In[85]:
def getArgs():
    restrfile = None
    thresh = 0.2
    if len(sys.argv) < 3:
        print("Not enough arguments.")
        exit(1)
    elif len(sys.argv) == 3:
        _, datafile, k = sys.argv
    elif len(sys.argv) == 4:
        _, datafile, k, thresh = sys.argv
    elif len(sys.argv) == 5:
        _, datafile, k, thresh, restrfile = sys.argv
    else:
        print("Usage: python3 validation.py <datafile.csv> <k> [threshold=0.2] [restrictions.txt]")
        print("(threshold necessary if giving restrictions)")
    
    df, tmp, isLabeled, attrs = readFiles(datafile,restrfile)
    
    return df, int(k), isLabeled, attrs, float(thresh)

def predict_kfold(df, numSplits, threshold, isLabeled, attrs):
    prev=None
    kfoldPreds = []
    cumOutput = None
    accuracies = []
    # all but one cross validation
    if numSplits == -1:
        numSplits = len(df)-1
    
    # split dataset kfold and generate predictions
    if numSplits <= 1:
        res, acc= classify(df, c45(df, attrs, threshold), asList=True, getAccuracy=True)
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
                tree=c45(trainingData, attrs, threshold)
                
                res, acc= classify(classifyData, tree, asList=True, getAccuracy=True)
                kfoldPreds += res
                accuracies.append(acc)
                prev=i
        
        trainingData = df[:prev]
        classifyData = df[prev:]
        res, acc= classify(classifyData, c45(trainingData, attrs, threshold), asList=True, getAccuracy=True)
        kfoldPreds += res
        accuracies.append(acc)
    
#     print("split accuracies:", accuracies, np.sum(accuracies))
    results = evaluate(df, kfoldPreds, asList=True)
    
    print("Average Accuracy:", np.sum(accuracies)/numSplits)
    for v in results:
        print(v, ":\n", results[v])
#     ret['actual'] = df.loc[:,df.columns[-1]:]
    
#     print()
#     print(f"-----Ran {numSplits}-fold cross-validation-----")
#     print("Overall Accuracy: ", accCorr[1]/len(ret))
#     print("Average Accuracy: ", accCorr[0]/numSplits)
#     print("\nConfusion Matrix: ")
#     print("Actual \u2193, Predicted \u2192")
#     print(confusion,'\n')
#     return ret


if __name__ == '__main__':
    df, k, isLabeled, attrs, thresh = getArgs()
    print(predict_kfold(df, k, thresh, isLabeled, attrs))